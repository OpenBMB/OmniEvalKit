import os
import logging
import warnings
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from o_e_Kit.utils.utils import uniform_sample
from o_e_Kit.utils.video_utils import concat_images_v3
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==================== Stack/Step 配置 完全无意义，用于提醒训练参数 ====================
# # 默认配置
# _DEFAULT_STACK_CONFIGS_BASE = {
#     0.5: [(1, 1)],       # 2FPS
#     0.2: [(1, 4)],       # 5FPS
#     0.1: [(1, 9)],       # 10FPS
#     0.05: [(1, 16)]      # 17FPS
# }

# # STACK_BALANCED 专属组合（n==m）
# _DEFAULT_STACK_CONFIGS_BALANCED = {
#     0.1: [(4, 4)],       # 8FPS
#     0.05: [(9, 9)]       # 18FPS
# }

# # FLEX_STACK 专属组合（n!=m 且 n!=1 且 m!=1）
# _DEFAULT_STACK_CONFIGS_FLEX = {
#     0.05: [(4, 9), (4, 16)]  # 13FPS, 20FPS
# }
class StackInfo(BaseModel):
    target_fps: int = Field(..., description="存在帧率, sum(nm_tuple)")
    nm_tuple: Tuple[int, ...] = Field(..., description="stack frame的参数, 任意长度 tuple")
    input_fps: int = Field(..., description="输入帧率,10或20, 原始视频抽取帧率")
    real_fps: int = Field(..., description="实际帧率, 非零元素个数, 模型实际上输入的帧数")

def load_video_and_stack_frames(
    video_path: str,
    start_sec: float = None,
    end_sec: float = None,
    fps: float = 10.0,
    nm_tuple: Tuple[int, ...] = (1, 0),
    sampling_mode: str = "fixed",
) -> Tuple[List[Image.Image], StackInfo]:
    """
    从视频中抽帧并进行 stack 处理。
    
    Args:
        video_path: 视频文件路径
        start_sec: 开始时间（秒），None 表示从头开始
        end_sec: 结束时间（秒），None 表示到视频结尾
        fps: 抽帧帧率，10 或 20
        nm_tuple: stack 参数，任意长度 tuple (如 (1,4), (4,4), (1,1,1,1,1))
        sampling_mode: "uniform" (均匀抽帧) 或 "fixed" (等分定格抽帧)，默认 "fixed"
    
    Returns:
        (stacked_frames, info): PIL.Image 列表和 StackInfo
    """
    frames = _extract_frames_ffmpeg(video_path, start_sec, end_sec, fps)
    stacked_frames, info = _apply_frame_stacking(frames, nm_tuple, int(fps), sampling_mode)
    return stacked_frames, info

def _extract_frames_ffmpeg(
    video_path: str,
    start_sec: float = None,
    end_sec: float = None,
    fps: float = 10.0,
) -> List[Image.Image]:
    """
    使用 ffmpeg 从视频中抽帧。
    
    Args:
        video_path: 视频文件路径
        start_sec: 开始时间（秒），None 表示从头开始
        end_sec: 结束时间（秒），None 表示到视频结尾
        fps: 抽帧帧率，支持 10 或 20
    
    Returns:
        PIL.Image 列表
    """
    
    if not os.path.exists(video_path):
        logger.warning(f"视频文件不存在: {video_path}")
        return []
    
    # 如果指定了时间范围，检查有效性
    if start_sec is not None and end_sec is not None:
        duration = end_sec - start_sec
        if duration <= 0:
            logger.warning(f"无效的时间范围: start={start_sec}, end={end_sec}")
            return []
    
    with tempfile.TemporaryDirectory(prefix="omni_frames_") as temp_dir:
        output_pattern = os.path.join(temp_dir, "frame_%05d.jpg")
        
        # 构建 ffmpeg 命令
        vf_filters = f"fps={fps}"
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
        ]
        
        # 添加开始时间（如果指定）
        if start_sec is not None:
            cmd.extend(["-ss", str(start_sec)])
        
        cmd.extend(["-i", video_path])
        
        # 添加持续时间（如果指定了结束时间）
        if start_sec is not None and end_sec is not None:
            cmd.extend(["-t", str(end_sec - start_sec)])
        elif end_sec is not None:
            # 只指定了结束时间，从头开始到 end_sec
            cmd.extend(["-t", str(end_sec)])
        
        cmd.extend([
            "-vf", vf_filters,
            "-q:v", "2",  # 高质量 JPEG
            "-y",
            output_pattern,
        ])
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg 抽帧失败: {e}, stderr: {e.stderr.decode(errors='ignore')}")
            return []
        
        # 读取生成的帧文件为 PIL.Image
        frame_files = sorted(Path(temp_dir).glob("frame_*.jpg"))
        
        frames = []
        for frame_path in frame_files:
            with Image.open(frame_path) as img:
                # 复制一份，因为退出 with 后文件会被关闭
                frames.append(img.convert("RGB").copy())
        
        return frames
        
def _apply_frame_stacking(
    frame_list: List[Image.Image],
    nm_tuple: tuple,
    input_fps: int,
    sampling_mode: str = "fixed"
) -> tuple:
    """
    帧抽取与堆叠，支持两种模式：
    
    1. 均匀抽帧 (uniform): 从整秒均匀采样 target_fps 帧，然后按 nm_tuple 分割
       - [1, 4] + 10fps → 均匀取 [0,2,4,6,8] → 分割 [0] + [2,4,6,8]
       
    2. 等分定格抽帧 (fixed): 按 real_fps 等分每秒，从每区开头取帧
       - [1, 4] + 10fps → 分 2 区 [0-4][5-9] → 取 [0] + [5,6,7,8] → 帧 1 + 帧 6789
    
    Args:
        frame_list: PIL.Image 列表
        nm_tuple: 任意长度的 tuple，每个元素表示该区域要取的帧数 (0 表示跳过)
        input_fps: 输入帧率，10 或 20
        sampling_mode: "uniform" 或 "fixed"，默认 "fixed"
    
    Returns:
        (stacked_frame_list, StackInfo)
    """
    if not frame_list:
        return [], StackInfo(target_fps=0, nm_tuple=nm_tuple, input_fps=input_fps, real_fps=0)
    
    assert input_fps == 10 or input_fps == 20, "input_fps must be 10 or 20"
    assert sampling_mode in ("uniform", "fixed"), "sampling_mode must be 'uniform' or 'fixed'"
    
    # real_fps = 非零元素的个数，决定每秒输出几个 segment
    real_fps = sum(1 for x in nm_tuple if x != 0)
    target_fps = sum(nm_tuple)  # 每秒需要的总帧数
    
    # 特殊情况：real_fps == 1，每秒只取一帧
    if real_fps == 1:
        sampled_frames = []
        frames_per_second = input_fps
        for sec_start in range(0, len(frame_list), frames_per_second):
            sec_frames = frame_list[sec_start:sec_start + frames_per_second]
            if sec_frames:
                sampled_frames.append(sec_frames[0])
        return sampled_frames, StackInfo(target_fps=target_fps, nm_tuple=nm_tuple, input_fps=input_fps, real_fps=real_fps)
    
    frames_per_second = input_fps
    stacked_frame_list = []
    
    for sec_start in range(0, len(frame_list), frames_per_second):
        sec_frames = frame_list[sec_start:sec_start + frames_per_second]
        if len(sec_frames) == 0:
            continue
        
        if sampling_mode == "uniform":
            # ========== 均匀抽帧模式 ==========
            # 从整秒均匀采样 target_fps 帧，然后按 nm_tuple 分割
            if len(sec_frames) >= target_fps:
                group = uniform_sample(sec_frames, target_fps)
            else:
                group = list(sec_frames)
                while len(group) < target_fps:
                    group.append(group[-1])
            
            # 按 nm_tuple 分割
            idx = 0
            for num_frames in nm_tuple:
                if num_frames == 0:
                    continue
                part = group[idx:idx + num_frames]
                idx += num_frames
                
                if num_frames == 1:
                    stacked_frame_list.append(part[0])
                else:
                    stacked_image = concat_images_v3(part)
                    stacked_frame_list.append(stacked_image)
        
        else:  # sampling_mode == "fixed"
            # ========== 等分定格抽帧模式 ==========
            # 按 real_fps 将每秒分成 real_fps 个等分区域
            zone_size = frames_per_second // real_fps
            
            zone_idx = 0
            for num_frames in nm_tuple:
                if num_frames == 0:
                    continue
                
                # 计算当前区域的帧范围
                zone_start = zone_idx * zone_size
                zone_end = zone_start + zone_size
                zone_frames = sec_frames[zone_start:zone_end]
                
                # 从区域开头取 num_frames 帧
                if len(zone_frames) >= num_frames:
                    part = zone_frames[:num_frames]
                else:
                    part = list(zone_frames)
                    while len(part) < num_frames:
                        part.append(part[-1] if part else sec_frames[-1])
                
                if num_frames == 1:
                    stacked_frame_list.append(part[0])
                else:
                    stacked_image = concat_images_v3(part)
                    stacked_frame_list.append(stacked_image)
                
                zone_idx += 1
    
    return stacked_frame_list, StackInfo(target_fps=target_fps, nm_tuple=nm_tuple, input_fps=input_fps, real_fps=real_fps)


# def has_usage_mode(usage: str, mode: str) -> bool:
#     """检查 usage 字符串中是否包含指定的模式"""
#     if not usage:
#         return False
#     return mode in usage
# ==================== Stack/Step 配置 ====================
# 可通过环境变量 STACK_CONFIG_YAML 指定 YAML 配置文件路径
#
# Stack 组合配置：step_sec -> 可选的 (n, m) 组合
# n+m 表示每秒分成2个 segment
# n=1: 第一个是 origin 单帧，第二个是 stack m帧
# n>1: 第一个是 stack n帧，第二个是 stack m帧
# 
# 基础组合（默认可选）：
#   step_sec=0.5  → 2FPS:  1+1
#   step_sec=0.2  → 5FPS:  1+4
#   step_sec=0.1  → 10FPS: 1+9
#   step_sec=0.05 → 17FPS: 1+16 (需要 HIGH_FPS_MODE)
#
# 特殊模式组合（需要对应 usage 才会加入候选）：
# STACK_BALANCED: 加入 n==m 的组合
#   - 0.1:  4+4  (8FPS)
#   - 0.05: 9+9  (18FPS)
# FLEX_STACK: 加入 n!=m 且 n!=1 且 m!=1 的组合
#   - 0.05: 4+9 (13FPS), 4+16 (20FPS)



# # 合法 step 值配置：input_fps -> (可选 step 列表, 最小 step 阈值)
# # 当 raw_step <= 阈值时从列表中随机选择，否则使用 max(最小低帧率step, raw_step)
# # 10fps: step=1(10fps), 2(5fps), 5(2fps), >=10(1fps或更低)
# # 20fps: step=1(20fps), 2(10fps), 4(5fps), 10(2fps), >=20(1fps或更低)
# _DEFAULT_STEP_CONFIGS = {
#     # 1:10fps, 2:5fps, 5:2fps
#     10.0: ([1, 2, 5], 5, 10),    # 可选 steps, 阈值, 最小低帧率 step 
#     # 1:20fps, 2:10fps, 4:5fps, 10:2fps
#     20.0: ([1, 2, 4, 10], 10, 20)
# }

# # 实际使用的配置（初始化为默认值，可被 YAML 覆盖）
# STACK_CONFIGS_BASE = dict(_DEFAULT_STACK_CONFIGS_BASE)
# STACK_CONFIGS_BALANCED = dict(_DEFAULT_STACK_CONFIGS_BALANCED)
# STACK_CONFIGS_FLEX = dict(_DEFAULT_STACK_CONFIGS_FLEX)
# STEP_CONFIGS = dict(_DEFAULT_STEP_CONFIGS)


# def _load_stack_configs_from_yaml():
#     """从环境变量指定的 YAML 文件加载配置，覆盖实际使用的配置"""
#     if yaml is None:
#         return  # yaml 未安装，使用默认配置
    
#     yaml_path = os.environ.get('STACK_CONFIG_YAML')
    
#     if not yaml_path or not os.path.exists(yaml_path):
#         return  # 不覆盖，使用默认配置
    
#     try:
#         with open(yaml_path, 'r') as f:
#             config = yaml.safe_load(f)
        
#         # 解析配置，将 list 转换为 tuple
#         def parse_stack_config(raw_config):
#             result = {}
#             for step_sec, combos in raw_config.items():
#                 result[float(step_sec)] = [tuple(c) for c in combos]
#             return result
        
#         def parse_step_config(raw_config):
#             result = {}
#             for fps, values in raw_config.items():
#                 steps, threshold, min_step = values
#                 result[float(fps)] = (steps, threshold, min_step)
#             return result
        
#         # 覆盖全局配置
#         global STACK_CONFIGS_BASE, STACK_CONFIGS_BALANCED, STACK_CONFIGS_FLEX, STEP_CONFIGS
        
#         if 'stack_configs_base' in config:
#             STACK_CONFIGS_BASE = parse_stack_config(config['stack_configs_base'])
#         if 'stack_configs_balanced' in config:
#             STACK_CONFIGS_BALANCED = parse_stack_config(config['stack_configs_balanced'])
#         if 'stack_configs_flex' in config:
#             STACK_CONFIGS_FLEX = parse_stack_config(config['stack_configs_flex'])
#         if 'step_configs' in config:
#             STEP_CONFIGS = parse_step_config(config['step_configs'])
        
#         logger.info(f"已从 {yaml_path} 加载 Stack/Step 配置")
        
#     except Exception as e:
#         logger.warning(f"加载 YAML 配置失败: {e}，使用默认配置")


# # 尝试从 YAML 加载配置（如果环境变量设置了）
# _load_stack_configs_from_yaml()

# def _select_stack_config(step_sec: float, usage: str) -> tuple:
#     """
#     根据 step_sec 和 usage 选择合适的 (n, m) stack 组合
    
#     Args:
#         step_sec: 每帧的时间间隔（秒），如 0.1, 0.05 等
#         usage: usage mode 字符串
    
#     Returns:
#         (n, m) 元组，表示每秒分成2个 segment：
#         - n=1: 第一个是 origin 单帧，第二个是 stack m帧
#         - n>1: 第一个是 stack n帧，第二个是 stack m帧
#     """
#     # 从基础配置开始
#     configs = list(STACK_CONFIGS_BASE.get(step_sec, [(1, 1)]))
    
#     # 如果启用 STACK_BALANCED，加入 balanced 专属组合
#     if has_usage_mode(usage, 'STACK_BALANCED'):
#         configs.extend(STACK_CONFIGS_BALANCED.get(step_sec, []))
    
#     # 如果启用 FLEX_STACK，加入 flex 专属组合
#     if has_usage_mode(usage, 'FLEX_STACK'):
#         configs.extend(STACK_CONFIGS_FLEX.get(step_sec, []))
    
#     return random.choice(configs)


def save_frames_to_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 2,
) -> bool:
    """
    将 PIL.Image 列表保存为视频。
    
    Args:
        frames: PIL.Image 列表
        output_path: 输出视频路径
        fps: 输出视频帧率
    
    Returns:
        是否成功
    """
    if not frames:
        logger.warning("帧列表为空，无法保存视频")
        return False
    
    with tempfile.TemporaryDirectory(prefix="video_save_") as temp_dir:
        # 先保存所有帧为图片
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
            frame.save(frame_path, "JPEG", quality=95)
        
        # 使用 ffmpeg 合成视频
        input_pattern = os.path.join(temp_dir, "frame_%05d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            output_path,
        ]
        
        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"保存视频失败: {e.stderr.decode(errors='ignore')}")
            return False


if __name__ == "__main__":
    import time
    
    video_path = "./example.mp4"
    output_dir = "./stack_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("测试 load_video_and_stack_frames")
    print("=" * 60)
    print(f"视频路径: {video_path}")
    print(f"输出目录: {output_dir}")
    
    # 测试1: 抽取前20秒，10fps，(1, 9) stack
    print("\n--- 测试1: 前20秒, 10fps, nm=(1,9) ---")
    start_time = time.time()
    stacked_frames, info = load_video_and_stack_frames(
        video_path,
        end_sec=20.0,
        fps=10,
        nm_tuple=(1, 9),
    )
    elapsed = time.time() - start_time
    print(f"抽帧+stack耗时: {elapsed:.2f}s")
    print(f"输出帧数: {len(stacked_frames)}")
    print(f"StackInfo: {info}")
    if stacked_frames:
        print(f"第一帧尺寸: {stacked_frames[0].size}")
    
    # 保存为视频，帧率为 real_fps
    output_video = os.path.join(output_dir, f"stacked_10fps_1_9_real{info.real_fps}fps.mp4")
    print(f"\n保存视频: {output_video} (fps={info.real_fps})")
    start_time = time.time()
    success = save_frames_to_video(stacked_frames, output_video, fps=info.real_fps)
    elapsed = time.time() - start_time
    print(f"保存视频耗时: {elapsed:.2f}s, 成功: {success}")
    
    # 测试2: 不同的 nm_tuple
    print("\n--- 测试2: 前20秒, 10fps, nm=(4,4) ---")
    stacked_frames, info = load_video_and_stack_frames(
        video_path,
        end_sec=20.0,
        fps=10,
        nm_tuple=(4, 4),
    )
    print(f"输出帧数: {len(stacked_frames)}")
    print(f"StackInfo: {info}")
    
    output_video = os.path.join(output_dir, f"stacked_10fps_4_4_real{info.real_fps}fps.mp4")
    print(f"保存视频: {output_video} (fps={info.real_fps})")
    success = save_frames_to_video(stacked_frames, output_video, fps=info.real_fps)
    print(f"保存成功: {success}")
    
    # 保存一些示例帧
    print("\n--- 保存示例帧 ---")
    for i, frame in enumerate(stacked_frames[:6]):
        save_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        frame.save(save_path, "JPEG", quality=95)
        print(f"保存: {save_path} ({frame.size})")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)