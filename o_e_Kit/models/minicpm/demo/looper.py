import torch
from typing import Optional, Union, Dict, List, Tuple
import librosa
from itertools import cycle
import numpy as np

class StreamProvider:
    """
    next_audio() ──> torch.Tensor  [S_chunk, H_a] 或 None
    next_image() ──> torch.Tensor  [H_i]         或 None
    """
    def __init__(
        self,
        audio_embeds: Optional[torch.Tensor] = None,   # [N_chunk, H_audio] # H_audio -> 1个位置     = 0.1s audio
        image_embeds: Optional[torch.Tensor] = None,   # [N_image, H_image] # H_image -> 整个视频    = 1 frame
        *,
        audio_chunk_len: int = 1, # 一次消耗  1个 H         
        image_step:     int = 64, # 一次消耗 64个 H
        audio_chunk_ms: int = 100, # 一次消耗 100ms audio
    ):
        # ---- 三条流 ----
        # Handle None or empty tensors gracefully
        self.audio = audio_embeds if audio_embeds is not None and len(audio_embeds) > 0 else None
        self.image = image_embeds if image_embeds is not None and len(image_embeds) > 0 else None

        # ---- 指针 ----
        self.ptr_a = 0
        self.ptr_i = 0

        # ---- 步长 ----
        self.audio_chunk_len = audio_chunk_len
        self.audio_chunk_ms = audio_chunk_ms
        self.image_step = image_step

    def next_audio(self) -> Optional[torch.Tensor]:
        if self.audio is None or self.ptr_a >= len(self.audio):
            return None
        chunk = self.audio[
            self.ptr_a : self.ptr_a + self.audio_chunk_len
        ]
        self.ptr_a += self.audio_chunk_len
        return chunk

    def next_image(self) -> Optional[torch.Tensor]:
        if self.image is None or self.ptr_i >= len(self.image):
            return None
        img = self.image[self.ptr_i : self.ptr_i + self.image_step]
        self.ptr_i += self.image_step
        return img

    def reset(self):
        self.ptr_a = self.ptr_i

    def finished(self):
        audio_finished = self.audio is None or self.ptr_a >= len(self.audio)
        image_finished = self.image is None or self.ptr_i >= len(self.image)
        return audio_finished and image_finished

    def print_progress(self):
        audio_len = len(self.audio) if self.audio is not None else 0
        image_len = len(self.image) if self.image is not None else 0
        print(f"StreamProvider - audio progress: {self.ptr_a}/{audio_len}, ptr_a: {self.ptr_a}, audio length: {audio_len}")
        print(f"StreamProvider - image progress: {self.ptr_i}/{image_len}, ptr_i: {self.ptr_i}, image length: {image_len}")
    
    def get_current_time(self):
        return self.ptr_a * self.audio_chunk_ms / 1000
        
class LoopPlanner:
    """
    next_plan()  →  (modal, token_id, action)
      modal ∈ {"sp_text", "image", "audio"}
      action ∈ {"prefill", "decode"}
    """
    def __init__(
        self,
        step_map: Dict[str, List[str]],
        loop: List[str],
        tokenizer,
        *,
        cycle_forever: bool = True,
        MAX_LOOP_TIME = 30
    ):
        # 1) 把字符串 token → id
        self.map_id: Dict[str, List[int]] = {
            k: [tokenizer.convert_tokens_to_ids(tok) for tok in v]
            for k, v in step_map.items()
        }
        # 2) 解析loop，提取token和action
        self.loop_items = []
        for item in loop:
            if ':' in item:
                token, action = item.split(':', 1)
                tid = tokenizer.convert_tokens_to_ids(token)
                self.loop_items.append((tid, action))
            else:
                tid = tokenizer.convert_tokens_to_ids(item)
                self.loop_items.append((tid, 'prefill'))
        
        self.iterator = cycle(self.loop_items) if cycle_forever else iter(self.loop_items)
        self.cycle_forever = cycle_forever
        # 反查字典：id → modal
        self.id2modal = {
            tid: modal for modal, tid_list in self.map_id.items() for tid in tid_list
        }

    def next_plan(self) -> Tuple[str, int, str]:
        tid, action = next(self.iterator)
        modal = self.id2modal.get(tid, "text")
        return modal, tid, action

    
    def reset(self):
        self.iterator = cycle(self.loop_items) if self.cycle_forever else iter(self.loop_items)


class MockStreamProvider:
    def __init__(self, frame_path, audio_path, processor, pad_audio=False, pad_frame=False):
        self.processor = processor
        self.frame_generator_list = list(self._build_frame_generator(frame_path)) if frame_path else []
        
        # 如果启用 pad_frame，将最后一帧重复 30 次
        if pad_frame and self.frame_generator_list:
            last_frame = self.frame_generator_list[-1]
            for _ in range(30):
                self.frame_generator_list.append(last_frame)
            print(f"Padded frames: added 30 repetitions of the last frame")
        
        self.audio_path = audio_path
        self.frame_idx = 0
        
        # Handle missing audio gracefully
        self.audio_wav = None
        self.sr = 16000  # Default sample rate
        if audio_path:
            try:
                print(f"Loading audio: {audio_path}")
                self.audio_wav, self.sr = librosa.load(audio_path, sr=16000, mono=True)
                print(f"Audio loaded: {self.audio_wav.shape}")
                if pad_audio and self.audio_wav is not None:
                    # 填充 30 秒的静音（30秒 * 16000 采样率）
                    silence_samples = 30 * self.sr  # 30 秒的静音
                    self.audio_wav = np.pad(self.audio_wav, (0, silence_samples), mode='constant')
                    print(f"Padded audio: added 30 seconds of silence")
                else:
                    # pad 到整秒
                    self.audio_wav = np.pad(self.audio_wav, (0, self.sr - len(self.audio_wav) % self.sr), mode='constant')
            except Exception as e:
                print(f"Warning: Failed to load audio from {audio_path}: {e}")
                self.audio_wav = None
        # 流式音频处理状态
        self.audio_chunk_idx = 0
        self.read_cursor_ms = 0
        self.CHUNK_MS = 100
        self.FIRST_CHUNK_MS = 135
        self.CNN_REDUNDANCY_MS = 20
        
        # 初始化流式Mel处理器
        if hasattr(processor, 'set_streaming_mode'):
            streaming_kwargs = dict(
                mode="exact",
                chunk_ms=self.CHUNK_MS,
                first_chunk_ms=self.FIRST_CHUNK_MS,
                cnn_redundancy_ms=self.CNN_REDUNDANCY_MS,
                # 滑窗参数（Trigger模式）
                enable_sliding_window=True,    # 启用滑窗
                slide_trigger_seconds=30.0,    # 当缓冲区达到30秒时触发滑窗
                slide_stride_seconds=10.0,     # 每次滑动10秒
            )
            # 兼容不同版本的 MiniCPMOProcessor：
            # - 新版本可能支持 verbose
            # - 旧版本不支持 verbose，会抛 TypeError
            try:
                processor.set_streaming_mode(verbose=False, **streaming_kwargs)
            except TypeError as e:
                if "unexpected keyword argument 'verbose'" in str(e):
                    processor.set_streaming_mode(**streaming_kwargs)
                else:
                    raise
            processor.reset_streaming()  # 关键：必须reset！

    def _build_frame_generator(self, frame_path):
        import os
        import re
        from PIL import Image
        # 支持常见的图像扩展名，并按文件名中的数字进行排序（如 frame_0004.jpg → 4）
        exts = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(frame_path) if f.lower().endswith(exts)]
        if not files:
            raise FileNotFoundError(f"No image frames found in {frame_path}. Supported extensions: {exts}")

        def extract_number(filename):
            name_without_ext = os.path.splitext(filename)[0]
            match = re.search(r"(\d+)", name_without_ext)
            return int(match.group(1)) if match else float("inf")

        for file in sorted(files, key=extract_number):
            print(f"Loading frame: {file}")
            frame = Image.open(os.path.join(frame_path, file))
            yield frame
    
    def next_frame(self, device=None):
        if self.frame_idx >= len(self.frame_generator_list):
            return None
        frame = self.frame_generator_list[self.frame_idx]
        self.frame_idx += 1
        data = self.processor.process_image([frame])
        if device is not None:
            data = data.to(device)
        return data
    
    def next_audio(self, device=None):
        """
        使用StreamingMelProcessorExact处理音频
        返回适合get_audio_embedding_streaming的数据格式
        """
        if self.audio_wav is None:
            raise ValueError("音频数据为空")
        
        # 计算需要读取的音频样本数
        need_samp = self.processor.get_streaming_chunk_size()
        start_samp = int(self.read_cursor_ms * self.sr / 1000)
        end_samp = start_samp + need_samp
        
        # 检查是否还有音频数据
        if start_samp >= len(self.audio_wav):
            raise ValueError("音频数据结束")
        
        # 提取音频chunk
        audio_chunk = self.audio_wav[start_samp:end_samp]
        
        if len(audio_chunk) < need_samp:
            raise ValueError(f"音频块长度 {len(audio_chunk)} < 期望长度 {need_samp}")
        
        # 使用StreamingMelProcessorExact处理，直接返回batch feature格式
        batch_feature = self.processor.process_audio_streaming(
            audio_chunk, 
            reset=False, 
            return_batch_feature=True  # 关键：返回batch feature格式
        )
        
        # 检查是否有输出
        if batch_feature is None or batch_feature.audio_features.shape[-1] == 0:
            raise ValueError("音频块长度为0")
        
        # 添加额外的元数据（供get_audio_embedding_streaming使用）
        batch_feature.chunk_idx = self.audio_chunk_idx
        batch_feature.use_extra_context = True
        batch_feature.prefix_extra_frames = 0 if self.audio_chunk_idx == 0 else 2
        batch_feature.suffix_extra_frames = 2
        
        # 更新读取位置（关键！）
        if self.audio_chunk_idx == 0:
            cfg = self.processor._streaming_mel_processor.get_config()
            self.read_cursor_ms = int(cfg.get('effective_first_chunk_ms', self.FIRST_CHUNK_MS))
        else:
            self.read_cursor_ms += self.CHUNK_MS
        
        self.audio_chunk_idx += 1
        
        # 移动到设备
        if device is not None:
            batch_feature = batch_feature.to(device)
        
        return batch_feature

    def all_data(self, device=None):
        """
        获取所有数据（用于离线处理）
        """
        # 音频：使用完整的音频波形
        audio_data = None
        if self.audio_wav is not None:
            audio_data = self.processor.process_audio([self.audio_wav], regroup_to_seconds=30, fps=100)
            if device is not None:
                audio_data = audio_data.to(device)
            
        # 图像：按帧聚合
        frame_data = None
        if self.frame_generator_list:
            frame_data = self.processor.process_image(self.frame_generator_list)
            if device is not None:
                frame_data = frame_data.to(device)
                
        return audio_data, frame_data
    
    def is_end(self):
        """
        检查是否已经处理完所有数据
        """
        # 音频结束：当读取位置超过音频长度
        audio_end = True
        if self.audio_wav is not None:
            start_samp = int(self.read_cursor_ms * self.sr / 1000)
            audio_end = start_samp >= len(self.audio_wav)
        
        # 图像结束：当索引超过列表长度
        frame_end = self.frame_idx >= len(self.frame_generator_list)
        
        return audio_end and frame_end
    
    def get_current_time(self):
        return self.read_cursor_ms / 1000
    
    def reset(self):
        """
        重置流式处理状态
        """
        self.frame_idx = 0
        self.audio_chunk_idx = 0
        self.read_cursor_ms = 0
        
        # 重置流式Mel处理器
        if hasattr(self.processor, 'reset_streaming'):
            self.processor.reset_streaming()
