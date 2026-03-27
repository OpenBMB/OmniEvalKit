"""
视频处理工具模块
包含StreamingBench相关的视频分割和处理功能
"""

import os
import ffmpeg
import tempfile
import subprocess
import base64
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


def parse_timestamp(timestamp: str) -> int:
    """
    将时间戳字符串转换为秒数
    Args:
        timestamp: 时间戳字符串，格式如 "00:03:10"
    Returns:
        int: 秒数
    """
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        return int(parts[0])


def split_video(video_path: str, start_time: int, end_time: int, output_dir: Optional[str] = None) -> str:
    """
    分割视频，提取指定时间段的视频片段
    
    Args:
        video_path: 原始视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_dir: 输出目录，如果为None则使用临时目录
    
    Returns:
        str: 分割后的视频文件路径
    """
    if output_dir is None:
        # 使用临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = temp_file.name
        temp_file.close()
    else:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"clip_{start_time}_{end_time}.mp4"
        output_path = os.path.join(output_dir, filename)
    
    try:
        # 使用ffmpeg进行视频分割
        duration = end_time - start_time
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(output_path, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        # 清理失败的输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"视频分割失败: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        # 清理失败的输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"视频分割失败: {str(e)}")


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    从视频中提取音频
    
    Args:
        video_path: 视频文件路径
        output_path: 音频输出路径，如果为None则使用临时文件
    
    Returns:
        str: 提取的音频文件路径
    """
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ar='16000', vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"音频提取失败: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"音频提取失败: {str(e)}")


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        dict: 包含视频信息的字典
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_info is None:
            raise ValueError(f"未能在文件 {video_path} 中找到视频流")
        
        return {
            'duration': float(video_info.get('duration', 0)),
            'width': int(video_info.get('width', 0)),
            'height': int(video_info.get('height', 0)),
            'fps': eval(video_info.get('r_frame_rate', '0/1'))
        }
        
    except Exception as e:
        raise Exception(f"获取视频信息失败: {str(e)}")


def cleanup_temp_video(file_path: str):
    """
    清理临时视频文件
    
    Args:
        file_path: 要删除的文件路径
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Warning: 无法删除临时文件 {file_path}: {e}") 


# modified from concat_images 因为 想试试 16 和 9 的组合
def concat_images_v3(images, bg_color=(255, 255, 255), cell_size=None,
                  line_color=(0, 0, 0), line_width=6):
    """
    images: List[PIL.Image.Image]
    规则：3 张 -> 1x3；4 张 -> 2x2；9 张 -> 3x3；其余：1xN
    仅在拼接处画分界线（不画外框）。
    """
    
    # 统一将输入转换为 PIL.Image：支持 PIL.Image、bytes/bytearray、base64 字符串
    _converted_images = []
    for im in images:
        if isinstance(im, Image.Image):
            _converted_images.append(im)
        elif isinstance(im, (bytes, bytearray)):
            _converted_images.append(Image.open(BytesIO(im)).convert("RGB"))
        elif isinstance(im, str):
            # 处理形如 'data:image/jpeg;base64,...' 或纯 base64
            b64 = im.split(',')[-1] if ';base64,' in im else im
            img_bytes = base64.b64decode(b64)
            _converted_images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
        else:
            raise TypeError(f"Unsupported image type: {type(im)}")
    images = _converted_images
    n = len(images)
    if n == 0:
        raise ValueError("images is empty")

    if n == 16:
        rows, cols = 4, 4
    elif n == 9:
        rows, cols = 3, 3
    elif n == 4:
        rows, cols = 2, 2
    elif n == 3:
        # 动态选择 1x3 / 3x1 / 2x2，使最终更接近正方形
        # 先用原图最大宽高确定单元格尺寸（下方 letterbox 会自适应）
        if cell_size is None:
            cell_w = max(im.width for im in images)
            cell_h = max(im.height for im in images)
        else:
            cell_w, cell_h = cell_size

        candidates = [(1, 3), (3, 1)]
        def canvas_ratio(r, c):
            W = c * cell_w + (c - 1) * line_width
            H = r * cell_h + (r - 1) * line_width
            return W / max(1, H)
        ratios = [abs(canvas_ratio(r, c) - 1.0) for (r, c) in candidates]
        best_idx = int(np.argmin(ratios))
        rows, cols = candidates[best_idx]
    elif n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # 动态选择 1x2 / 2x1，使最终更接近正方形
        if cell_size is None:
            cell_w = max(im.width for im in images)
            cell_h = max(im.height for im in images)
        else:
            cell_w, cell_h = cell_size
        candidates = [(1, 2), (2, 1)]
        def canvas_ratio(r, c):
            W = c * cell_w + (c - 1) * line_width
            H = r * cell_h + (r - 1) * line_width
            return W / max(1, H)
        ratios = [abs(canvas_ratio(r, c) - 1.0) for (r, c) in candidates]
        # 如出现并列，依据平均宽高比进行决策：横向排列适合横图，纵向排列适合竖图
        if ratios[0] == ratios[1]:
            avg_ar = np.mean([im.width / max(1, im.height) for im in images])
            rows, cols = (1, 2) if avg_ar >= 1.0 else (2, 1)
        else:
            best_idx = int(np.argmin(ratios))
            rows, cols = candidates[best_idx]
    else:
        rows, cols = 1, n

    # 单元格尺寸
    if cell_size is None:
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
    else:
        cell_w, cell_h = cell_size

    # 保持纵横比缩放到单元格
    def letterbox(im, tw, th):
        im = im.convert("RGB")
        w, h = im.size
        s = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        try:
            im_r = im.resize((nw, nh), Image.Resampling.BICUBIC)
        except AttributeError:
            im_r = im.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (tw, th), bg_color)
        canvas.paste(im_r, ((tw - nw) // 2, (th - nh) // 2))
        return canvas

    # 仅在内部缝隙处留出 line_width 的带状区域作为分界线
    W = cols * cell_w + (cols - 1) * line_width
    H = rows * cell_h + (rows - 1) * line_width
    canvas = Image.new("RGB", (W, H), line_color)

    for i, im in enumerate(images[:rows * cols]):
        r, c = divmod(i, cols)
        cell = letterbox(im, cell_w, cell_h)
        x = c * (cell_w + line_width)
        y = r * (cell_h + line_width)
        canvas.paste(cell, (x, y))

    return canvas