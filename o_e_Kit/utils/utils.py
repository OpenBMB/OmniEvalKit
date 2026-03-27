from pydub import AudioSegment
from PIL import Image
import librosa
import wave
import contextlib
import os
import mmap
import numpy as np
import warnings
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Tuple

import torch
import logging

# pyrubberband 用于高质量音频变速
try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except ImportError:
    VideoReader = None
    cpu = None
    _HAS_DECORD = False

try:
    import pyrubberband as pyrb
    _HAS_PYRUBBERBAND = True
except ImportError:
    pyrb = None
    _HAS_PYRUBBERBAND = False

logger = logging.getLogger(__name__)

try:
    # 优先使用 torchvision 作为备用视频解码后端
    from torchvision import io as tv_io

    _HAS_TORCHVISION = True
except Exception:
    tv_io = None
    _HAS_TORCHVISION = False

def read_mp3_file_to_wav_bytes(mp3_file_path: str) -> bytes:
    audio_segment = AudioSegment.from_file(mp3_file_path)
    return audio_segment.export(format="wav").read()


def uniform_sample(l, n):
    """均匀采样函数"""
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def _sample_video_frame_indices(
    vr: "VideoReader",
    max_num_frames: int,
    max_fps: float,
) -> tuple[List[int], List[float], float]:
    """
    根据示例逻辑采样视频帧，并返回对应的时间戳：
    - 若 duration > max_num_frames: 先以 0.1 秒粒度生成 timestamps & frame_idx，
      再用 uniform_sample 均匀采样到 max_num_frames 个点；
    - 否则：按 1fps 抽帧，此时 timestamps 是 0,1,2,...

    返回：
        frame_idx:   采样后的帧下标列表
        timestamps:  每个帧对应的时间戳（秒）
        avg_fps:     视频平均帧率
    """
    avg_fps = float(vr.get_avg_fps())
    num_total_frames = len(vr)
    duration = num_total_frames / avg_fps  # 总时长（秒）

    if duration > max_num_frames:
        # 长视频：以 0.1 秒粒度生成时间戳，再均匀采样到 max_num_frames
        step = 0.1
        num_steps = int(duration / step)
        timestamps = [round(i * step, 1) for i in range(num_steps)]
        frame_idx = [
            min(int(ts * avg_fps), num_total_frames - 1) for ts in timestamps
        ]
        if len(frame_idx) > max_num_frames:
            frame_idx = uniform_sample(frame_idx, max_num_frames)
            timestamps = uniform_sample(timestamps, max_num_frames)
    else:
        # 短视频：固定按 1fps 抽帧，timestamps 为 0,1,2,...
        int_duration = int(duration)
        frame_idx = [int(i * avg_fps) for i in range(int_duration)]
        timestamps = [float(i) for i in range(int_duration)]

    return frame_idx, timestamps, avg_fps


def _load_video_frames_with_decord(
    video_path: str,
    max_num_frames: int,
    max_fps: float,
) -> tuple[list[Image.Image], list[int], list[float], float, int]:
    """
    使用 decord 加载视频帧。

    返回:
        frames:      PIL.Image 列表
        frame_idx:   帧下标列表
        timestamps:  每帧对应的时间戳（秒）
        avg_fps:     视频平均帧率
        total_frames:视频总帧数
    失败时返回 ([], [], [], 0.0, 0)
    """
    if not _HAS_DECORD:
        return [], [], [], 0.0, 0

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        warnings.warn(
            f"[video_decord] 无法打开视频: {video_path}, error: {e}"
        )
        return [], [], [], 0.0, 0

    try:
        frame_idx, timestamps, avg_fps = _sample_video_frame_indices(
            vr, max_num_frames=max_num_frames, max_fps=float(max_fps)
        )
        if not frame_idx:
            return [], [], [], avg_fps, len(vr)
        frames_np = vr.get_batch(frame_idx).asnumpy()
        frames = [
            Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames_np
        ]
        total_frames = len(vr)
        return frames, frame_idx, timestamps, avg_fps, total_frames
    except Exception as e:
        warnings.warn(
            f"[video_decord] 解码视频帧失败: {video_path}, error: {e}"
        )
        return [], [], [], 0.0, 0


def _load_video_frames_with_torchvision(
    video_path: str,
    max_num_frames: int,
    max_fps: float,
) -> tuple[list[Image.Image], list[int], list[float], float, int]:
    """
    使用 torchvision.io.read_video 加载视频帧，作为 decord 的回退方案。

    返回:
        frames:      PIL.Image 列表
        frame_idx:   帧下标列表
        timestamps:  每帧对应的时间戳（秒）
        avg_fps:     视频平均帧率
        total_frames:视频总帧数
    失败时返回 ([], [], [], 0.0, 0)
    """
    if not _HAS_TORCHVISION:
        warnings.warn(
            f"[video_torchvision] torchvision 不可用，无法作为回退解码器: {video_path}"
        )
        return [], [], [], 0.0, 0

    try:
        with torch.no_grad():
            video, audio, info = tv_io.read_video(
                video_path,
                start_pts=0.0,
                end_pts=None,
                pts_unit="sec",
                output_format="TCHW",
            )
    except Exception as e:
        warnings.warn(
            f"[video_torchvision] read_video 失败: {video_path}, error: {e}"
        )
        return [], [], [], 0.0, 0

    total_frames = int(video.size(0))
    if total_frames == 0:
        return [], [], [], 0.0, 0

    video_fps = float(info.get("video_fps", 0.0) or 0.0)
    if video_fps <= 0:
        # 退化处理：如果拿不到 fps，则按 25fps 估计
        video_fps = 25.0

    duration = total_frames / video_fps
    # 根据 max_fps 和 max_num_frames 计算目标帧数
    if max_fps > 0:
        n_by_fps = int(duration * max_fps + 0.5)
        nframes = max(1, min(max_num_frames, total_frames, n_by_fps or max_num_frames))
    else:
        nframes = max(1, min(max_num_frames, total_frames))

    with torch.no_grad():
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        # video: (T, C, H, W)
        sampled = video[idx]  # (nframes, C, H, W)
        # 转为 numpy，再转 PIL
        frames_np = (
            sampled.permute(0, 2, 3, 1)  # (nframes, H, W, C)
            .contiguous()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    frames: list[Image.Image] = [
        Image.fromarray(v).convert("RGB") for v in frames_np
    ]
    frame_idx = idx.tolist()
    timestamps = [fi / video_fps for fi in frame_idx]
    avg_fps = video_fps
    return frames, frame_idx, timestamps, avg_fps, total_frames


def _load_video_frames_with_ffmpeg(
    video_path: str,
    max_num_frames: int,
    max_fps: float,
) -> tuple[list[Image.Image], list[int], list[float], float, int]:
    """
    使用 FFmpeg 命令行抽帧，作为 decord 的第二层回退方案。

    简化版实现：
        - 通过 `-vf fps={max_fps}` 控制时间采样频率；
        - 通过 `-vframes max_num_frames` 控制最多抽取帧数；
        - 将输出帧保存为临时 JPEG 文件，再读回为 PIL.Image。

    返回:
        frames:      PIL.Image 列表
        frame_idx:   帧下标列表（相对于抽取序列的索引 0..N-1）
        timestamps:  每帧对应的时间戳（秒，按 fps 近似计算）
        avg_fps:     近似帧率（即 max_fps）
        total_frames:抽出的总帧数
    """
    if max_num_frames <= 0:
        max_num_frames = 1
    target_fps = float(max_fps) if max_fps and max_fps > 0 else 1.0

    if not os.path.exists(video_path):
        warnings.warn(f"[video_ffmpeg] 视频不存在: {video_path}")
        return [], [], [], 0.0, 0

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="ffmpeg_frames_")

    try:
        output_pattern = os.path.join(temp_dir_obj.name, "frame_%05d.jpg")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps={target_fps}",
            "-q:v",
            "2",
            "-vframes",
            str(max_num_frames),
            output_pattern,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            warnings.warn(
                "[video_ffmpeg] 系统中未找到 ffmpeg，可安装 ffmpeg 或让环境 PATH 可见。"
            )
            return [], [], [], 0.0, 0
        except subprocess.CalledProcessError as e:
            warnings.warn(f"[video_ffmpeg] ffmpeg 抽帧失败: {video_path}, error: {e}")
            return [], [], [], 0.0, 0

        frame_files = sorted(Path(temp_dir_obj.name).glob("frame_*.jpg"))
        if not frame_files:
            return [], [], [], 0.0, 0

        frames: list[Image.Image] = []
        frame_idx: list[int] = []
        timestamps: list[float] = []

        for idx, frame_path in enumerate(frame_files):
            try:
                with Image.open(frame_path) as img:
                    frame = img.convert("RGB")
            except Exception as e:
                warnings.warn(
                    f"[video_ffmpeg] 读取中间帧失败: {frame_path}, error: {e}"
                )
                continue

            frames.append(frame)
            frame_idx.append(idx)
            # 近似时间戳：按 target_fps 均匀采样
            timestamps.append(idx / target_fps)

        total_frames = len(frames)
        if total_frames == 0:
            return [], [], [], 0.0, 0

        avg_fps = target_fps
        return frames, frame_idx, timestamps, avg_fps, total_frames

    finally:
        temp_dir_obj.cleanup()


def encode_video(video_path, MAX_NUM_FRAMES=64, MAX_FPS=1):
    """
    根据视频FPS智能采样帧，最多采样MAX_NUM_FRAMES帧。
    优先使用 decord，其次尝试 FFmpeg，最后回退到 torchvision.io.read_video。
    """
    # 1) 首先尝试 decord
    frames, frame_idx, timestamps, avg_fps, total_frames = _load_video_frames_with_decord(
        video_path, MAX_NUM_FRAMES, float(MAX_FPS)
    )
    backend = "decord"

    # 2) 如 decord 失败或无帧，则尝试 FFmpeg
    if not frames:
        frames, frame_idx, timestamps, avg_fps, total_frames = _load_video_frames_with_ffmpeg(
            video_path, MAX_NUM_FRAMES, float(MAX_FPS)
        )
        backend = "ffmpeg" if frames else backend

    # 3) 如 FFmpeg 仍失败，再尝试 torchvision
    if not frames:
        frames, frame_idx, timestamps, avg_fps, total_frames = _load_video_frames_with_torchvision(
            video_path, MAX_NUM_FRAMES, float(MAX_FPS)
        )
        backend = "torchvision" if frames else "none"

    if not frames:
        warnings.warn(f"[encode_video] 无法从视频中解码任何帧: {video_path}")
        return []

    # 采样 FPS（仅用于日志）
    sample_fps = 0.0
    if total_frames > 0 and avg_fps > 0 and frame_idx:
        sample_fps = len(frame_idx) / max(total_frames, 1e-6) * avg_fps

    logger.info(
        f"Video: {video_path}, backend={backend}, sample_fps={sample_fps:.3f}, "
        f"num frames: {len(frames)}, raw_len: {total_frames}, raw_fps={avg_fps:.3f}"
    )
    return frames


def get_audio_duration(audio_path: str) -> Optional[float]:
    """
    获取音频文件的时长（秒）
    对于 WAV 文件优先使用快速的头部读取方法，失败时回退到 librosa
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        音频时长（秒），如果读取失败返回 None
    """
    # 如果是 WAV 文件，先尝试使用快速方法
    if audio_path.lower().endswith('.wav'):
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    
    # 其他格式或特殊情况，直接使用 librosa
    return librosa.get_duration(filename=audio_path)


def adjust_audio_to_duration(
    waveform: np.ndarray,
    sr: int = 16000,
    target_seconds: float = 0.0,
) -> np.ndarray:
    """
    将音频“压缩/加速”到指定时长（秒）附近。
    
    - 只在当前时长大于 target_seconds 时进行加速；
    - 否则保留原始音频。
    
    Args:
        waveform: 音频波形，一维 numpy 数组
        sr: 采样率，默认 16kHz
        target_seconds: 目标时长（秒）
    """
    if waveform is None:
        return None
    if target_seconds is None or target_seconds <= 0:
        return waveform
    
    cur_len = waveform.shape[0]
    cur_dur = cur_len / float(sr)
    
    # 短于目标长度则不做处理
    if cur_dur <= target_seconds:
        return waveform
    
    rate = cur_dur / target_seconds  # >1 表示加速
    try:
        return librosa.effects.time_stretch(waveform, rate=rate)
    except Exception:
        # time_stretch 失败时退化为简单截断
        max_audio_samples = int(sr * target_seconds)
        if max_audio_samples > 0 and waveform.shape[0] > max_audio_samples:
            return waveform[:max_audio_samples]
        return waveform

# ==================== 媒体加载工具函数 ====================

def load_image(image_path: str) -> Image.Image:
    """加载图片，返回 PIL.Image"""
    return Image.open(image_path).convert("RGB")

def load_audio(wav_path: str, sr: int = 16000, speed: float = 1.0, trim_end: float = 0.0) -> np.ndarray:
    """
    加载音频文件，返回指定采样率的波形
    
    Args:
        wav_path: 音频文件路径
        sr: 目标采样率，默认 16kHz
        speed: 变速倍数，>1.0 加速（用于 OOM 重试）
        trim_end: 截取掉末尾多少秒（用于 OOM 重试）
    
    Returns:
        waveform: numpy array
    """
    try:
        # 使用 mmap 加速大文件读取
        with open(wav_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                waveform = librosa.load(m, sr=sr)[0]
    except Exception as e:
        # 回退到普通加载
        waveform = librosa.load(wav_path, sr=sr)[0]
    
    # 变速处理
    if speed > 1.0:
        waveform = speedup_audio(waveform, sr=sr, speed=speed)
    
    # 截断处理
    if trim_end > 0:
        waveform = trim_audio_segment(waveform, sr=sr, trim_end=trim_end)
    
    return waveform


def speedup_audio(waveform: np.ndarray, sr: int = 16000, speed: float = 1.0) -> np.ndarray:
    """
    对音频进行变速处理（保持音高不变）
    
    Args:
        waveform: 音频波形 numpy array
        sr: 采样率
        speed: 变速倍数，>1.0 加速，<1.0 减速
    
    Returns:
        变速后的音频波形
    """
    if speed == 1.0:
        return waveform
    
    if _HAS_PYRUBBERBAND:
        # 使用 pyrubberband 高质量变速（保持音高）
        try:
            return pyrb.time_stretch(waveform, sr, speed)
        except Exception as e:
            warnings.warn(f"[speedup_audio] pyrubberband failed: {e}, fallback to resample")
    
    # 回退方案：使用 librosa resample（会改变音高，但足够用于 OOM 回退）
    # 加速 = 重采样到更低采样率后再恢复
    target_len = int(len(waveform) / speed)
    return librosa.resample(waveform, orig_sr=len(waveform), target_sr=target_len)


def speedup_content_audio(content: List[Any], speed: float = 1.0, sr: int = 16000) -> Tuple[List[Any], int]:
    """
    对 content 列表中的所有音频进行变速处理
    
    Args:
        content: 包含 Image/ndarray/str 的内容列表
        speed: 变速倍数
        sr: 音频采样率
    
    Returns:
        (new_content, audio_count): 变速后的 content 和处理的音频数量
    """
    if speed == 1.0:
        return content, 0
    
    new_content = []
    audio_count = 0
    for item in content:
        if isinstance(item, np.ndarray):
            new_content.append(speedup_audio(item, sr=sr, speed=speed))
            audio_count += 1
        else:
            new_content.append(item)
    
    return new_content, audio_count


def load_video(video_path: str, max_frames: int = 16, max_fps: float = 1.0) -> List[Image.Image]:
    """
    加载视频，返回 PIL.Image 帧列表
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数
        max_fps: 目标采样率（帧/秒），默认 1.0
    
    Returns:
        frames: PIL.Image 对象列表
    """
    frames = encode_video(video_path, MAX_NUM_FRAMES=max_frames, MAX_FPS=max_fps)
    return frames if frames else []


def load_video_and_audio(video_path: str, max_frames: int = 16, audio_sr: int = 16000, max_fps: float = 1.0, 
                         interleave_fps: float = 1.0, audio_speed: float = 1.0, audio_trim_end: float = 0.0) -> tuple:
    """
    加载视频帧 + 从同路径查找并加载音频
    
    音频查找规则：
    - 将视频文件扩展名替换为 .wav, .mp3, .m4a, .flac
    - 按顺序查找，找到第一个存在的文件
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大视频帧数
        audio_sr: 音频采样率
        max_fps: 视频采样率（帧/秒），默认 1.0
        audio_speed: 音频变速倍数，>1.0 加速（用于 OOM 重试）
        audio_trim_end: 截取掉末尾多少秒（用于 OOM 重试）
    
    Returns:
        (frames, waveform) 元组

    约定：
    - 视频帧数由 max_frames 和 max_fps 控制（见 load_video / encode_video）；
    - 音频会被「压缩/加速」到与视频时间窗口大致对齐：
      复用 adjust_audio_to_duration 的逻辑（只在音频更长时加速/截断）。
    """
    # 加载视频帧（统一走 encode_video / _sample_video_frame_indices 逻辑）
    frames = load_video(video_path, max_frames=max_frames, max_fps=max_fps)
    
    # 加载音频：统一复用 _load_waveform_and_duration 的查找 / 回退逻辑
    waveform, _ = _load_waveform_and_duration(video_path, audio_sr=audio_sr)
    if waveform is None:
        warnings.warn(f"[load_av] 未能为视频加载音频: {video_path}")

    # 音频变速处理（用于 OOM 重试）
    if waveform is not None:
        if audio_speed > 1.0:
            waveform = speedup_audio(waveform, sr=audio_sr, speed=audio_speed)
        # 音频截断处理
        if audio_trim_end > 0:
            waveform = trim_audio_segment(waveform, sr=audio_sr, trim_end=audio_trim_end)
        # 调试输出音频长度
        info = f"音频长度: {waveform.shape[0] / audio_sr:.1f}s"
        if audio_speed > 1.0:
            info += f" (speed={audio_speed}x)"
        if audio_trim_end > 0:
            info += f" (trim_end={audio_trim_end}s)"
        print(info)
    else:
        # 无音频时，不中断流程，后续将只使用视频帧
        print("[load_av] 当前样本未加载到音频，仅使用视频帧进行推理。")

    return frames, waveform


def _load_waveform_and_duration(
    video_path: str,
    audio_sr: int,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    加载与视频路径匹配的音频波形及其时长（秒）。
    优先查找同名音频文件，其次从视频轨中提取。
    """
    base_path = os.path.splitext(video_path)[0]
    audio_path = None
    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        candidate = base_path + ext
        if os.path.exists(candidate):
            audio_path = candidate
            break

    waveform: Optional[np.ndarray] = None
    audio_duration: Optional[float] = None

    if audio_path:
        try:
            waveform = load_audio(audio_path, sr=audio_sr)
            audio_duration = waveform.shape[0] / float(audio_sr)
        except Exception as e:
            warnings.warn(f"[load_waveform] 外部音频加载失败: {audio_path}, error: {e}")


    # 如果 audio_path 没有，那就和训练时候一样去加一个长度相同与视频片段的 音频背景声
    if waveform is None:
        return None, None

    return waveform, audio_duration


def get_video_frame_audio_segments(
    video_path: str,
    max_frames: int = 16,
    max_fps: float = 1.0,
    audio_sr: int = 16000,
) -> tuple[List[Image.Image], List[np.ndarray], List[int], List[float]]:
    """
    基于时间戳的音视频对齐采样：
    - 在时间轴上采样最多 max_frames 个时间点（且采样频率不超过 max_fps）；
    - 同时采样对应的视频帧；
    - 按相邻时间戳 [t_i, t_{i+1}) / [t_last, audio_end) 切分音频。
    
    返回：
        frames:        对齐后的视频帧列表
        audio_segments:对齐后的音频片段列表（与 frames 一一对应或更短）
        frame_idx:     每个帧在原视频中的下标
        timestamps:    每个帧对应的时间戳（秒）
    """
    # 1. 优先使用 decord 加载视频帧
    frames, frame_idx, timestamps, avg_fps, total_frames = _load_video_frames_with_decord(
        str(video_path), max_frames, float(max_fps)
    )
    backend = "decord"

    # 2. 如 decord 失败或未返回帧，则尝试 FFmpeg
    if not frames:
        frames, frame_idx, timestamps, avg_fps, total_frames = _load_video_frames_with_ffmpeg(
            str(video_path), max_frames, float(max_fps)
        )
        backend = "ffmpeg" if frames else backend

    # 3. 如 FFmpeg 仍失败，再尝试 torchvision
    if not frames:
        frames, frame_idx, timestamps, avg_fps, total_frames = _load_video_frames_with_torchvision(
            str(video_path), max_frames, float(max_fps)
        )
        backend = "torchvision" if frames else "none"

    if not frames:
        warnings.warn(
            f"[get_video_frame_audio_segments] 无法为视频加载任何帧: {video_path}, backend={backend}"
        )
        return [], [], [], []

    # 3. 加载音频
    waveform, audio_duration = _load_waveform_and_duration(video_path, audio_sr=audio_sr)
    if waveform is None or audio_duration is None:
        # 无音频则仅返回视频帧
        return frames, [], frame_idx, timestamps

    # 6. 使用 timestamps 切分音频
    audio_segments: List[np.ndarray] = []
    for i, start_time in enumerate(timestamps):
        if i < len(timestamps) - 1:
            end_time = timestamps[i + 1]
        else:
            # 最后一段延伸到音频结束（以音频自身时长为准）
            end_time = audio_duration

        start_sample = max(0, int(start_time * audio_sr))
        end_sample = max(start_sample, int(end_time * audio_sr))
        segment = waveform[start_sample:end_sample]
        if segment.size == 0:
            continue
        audio_segments.append(segment)

    # 7. 对齐长度并返回
    num_pairs = min(len(frames), len(audio_segments))
    frames = frames[:num_pairs]
    audio_segments = audio_segments[:num_pairs]
    frame_idx = frame_idx[:num_pairs]
    timestamps = timestamps[:num_pairs]

    print(f"[get_video_frame_audio_segments] video_path={video_path}, frames={len(frames)}, audio_segments={len(audio_segments)}, frame_idx={len(frame_idx)}, timestamps={len(timestamps)}, backend={backend}")
    logger.info(f"[get_video_frame_audio_segments] video_path={video_path}, frames={len(frames)}, audio_segments={len(audio_segments)}, frame_idx={len(frame_idx)}, timestamps={len(timestamps)}, backend={backend}")
    return frames, audio_segments, frame_idx, timestamps


def trim_audio_segment(waveform: np.ndarray, sr: int, trim_end: float) -> np.ndarray:
    """
    截断音频末尾
    
    Args:
        waveform: 音频波形
        sr: 采样率
        trim_end: 截取掉末尾多少秒
    
    Returns:
        截断后的音频（至少保留 0.1s）
    """
    if trim_end <= 0:
        return waveform
    
    samples_to_trim = int(trim_end * sr)
    min_samples = int(0.1 * sr)  # 至少保留 0.1s
    
    if len(waveform) - samples_to_trim >= min_samples:
        return waveform[:-samples_to_trim]
    elif len(waveform) > min_samples:
        return waveform[:min_samples]
    else:
        return waveform


def load_video_and_audio_interleaved(
    video_path: str,
    max_frames: int = 16,
    max_fps: float = 1.0,
    audio_sr: int = 16000,
    audio_speed: float = 1.0,
    audio_trim_end: float = 0.0,
) -> List[Any]:
    """
    基于时间戳的音视频交错加载：
    直接返回 [frame_1, audio_seg_1, frame_2, audio_seg_2, ...] 的 media 列表。
    在 OMNI_DEBUG_AV_INTERLEAVE=1 时，会在正常评测流程中打印调试信息。
    
    Args:
        audio_speed: 音频变速倍数，>1.0 加速（用于 OOM 重试时压缩音频长度）
        audio_trim_end: 每段音频截取掉末尾多少秒（用于 OOM 重试时减少音频长度）
    """
    frames, audio_segments, frame_idx, timestamps = get_video_frame_audio_segments(
        video_path=video_path,
        max_frames=max_frames,
        max_fps=max_fps,
        audio_sr=audio_sr,
    )
    
    # 音频变速处理（用于 OOM 重试）
    if audio_speed > 1.0 and audio_segments:
        audio_segments = [speedup_audio(seg, sr=audio_sr, speed=audio_speed) for seg in audio_segments]
    
    # 音频截断处理（用于 OOM 重试）
    if audio_trim_end > 0 and audio_segments:
        audio_segments = [trim_audio_segment(seg, sr=audio_sr, trim_end=audio_trim_end) for seg in audio_segments]

    # 调试输出：仅当环境变量开启时生效
    if os.getenv("OMNI_DEBUG_AV_INTERLEAVE", "0") == "1":
        num_frames = len(frames)
        num_segments = len(audio_segments)
        print(f"[debug_interleave] video_path={video_path}, audio_speed={audio_speed}")
        print(f"[debug_interleave] frames={num_frames}, audio_segments={num_segments}")
        if audio_segments:
            total_samples = sum(seg.shape[0] for seg in audio_segments)
            total_dur = total_samples / float(audio_sr)
            print(f"[debug_interleave] sum(audio_segments_dur) ≈ {total_dur:.3f}s (sr={audio_sr})")

            n = min(5, num_segments)
            print(f"[debug_interleave] 展示前 {n} 段：")
            for i in range(n):
                seg = audio_segments[i]
                seg_dur = seg.shape[0] / float(audio_sr)
                start_t = timestamps[i] if i < len(timestamps) else None
                end_t = timestamps[i + 1] if i + 1 < len(timestamps) else None
                print(
                    f"  [{i}] frame_idx={frame_idx[i] if i < len(frame_idx) else 'NA'}, "
                    f"t=[{start_t}, {end_t if end_t is not None else 'end'}), "
                    f"seg_dur={seg_dur:.3f}s, samples={seg.shape[0]}"
                )

    if not frames:
        return []
    if not audio_segments:
        # 没有音频时，退化为仅返回帧
        return frames

    media: List[Any] = []
    for f, a in zip(frames, audio_segments):
        media.append(f)
        media.append(a)
    return media