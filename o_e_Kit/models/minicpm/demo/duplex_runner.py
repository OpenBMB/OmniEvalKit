'''
@File    :   duplex_runner.py
@Time    :   2025/11/12 11:11:11
@Desc    :   DuplexRunner类，用于运行Duplex模型
双工模型，输入 视频流 和 音频流，并产生实时的文本回复
scheme： 
speak: <unit><image><unk></image><audio>x10<speak>xxxx<chunk_eos></unit>
listen: <unit><image><unk></image><audio>x10<listen></unit>
end of turn: <unit><image><unk></image><audio>x10<speak>xxxx<turn_eos><chunk_eos></unit>
'''
import os
import subprocess
import torch
import json
from transformers import AutoTokenizer
from typing import Generator, Optional

try:
    from decoder import StreamDecoder
    from looper import LoopPlanner, StreamProvider, MockStreamProvider
except ImportError:
    from o_e_Kit.models.minicpm.demo.decoder import StreamDecoder
    from o_e_Kit.models.minicpm.demo.looper import LoopPlanner, StreamProvider, MockStreamProvider

from o_e_Kit.utils.stack_utils import load_video_and_stack_frames

# Default duplex config path
DEFAULT_DUPLEX_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../configs/duplex_configs.json"
)

def load_duplex_configs(config_path: Optional[str] = None) -> dict:
    """Load and flatten duplex configurations."""
    if config_path is None:
        config_path = DEFAULT_DUPLEX_CONFIG_PATH
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using default duplex configs")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    # Flatten the config structure for easier access
    merged = {}
    for category in full_config.values():
        for subcategory in category.values():
            merged.update(subcategory)
    return merged

class Duplex:
    def __init__(self, model, tokenizer, device='cuda', processor=None, ls_mode='explicit', 
                 duplex_config_path: Optional[str] = None):
        self.m = model # 整个 模型
        self.tok = tokenizer
        self.decoder = StreamDecoder(llm=self.m.llm, tokenizer=self.tok) # llm 部分做了一个小的封装
        self.processor = processor
        self.dev = device
        self.step_map = {
            "sp_text": ["<unit>", "</unit>", "<image>", "</image>", "<|listen|>"],
            "image": ["<unk>"],
            "audio": ["<|audio|>"],
        }
        self.loop = []
        self.ls_mode = ls_mode

        # FPS 配置：通过环境变量 DUPLEX_FPS 设置，默认为 1
        self.fps = int(os.environ.get("DUPLEX_FPS", "1"))
        
        # Stack 配置：nm_tuple = (n, m)，real_fps = len([x for x in nm_tuple if x != 0])
        # 默认 (1, 0) 表示不使用 stack 模式
        self.nm_tuple = (1, 0)
        self.real_fps = 1  # 实际每秒输出的帧数（segment 数）

        # Load duplex configs
        self.duplex_configs = load_duplex_configs(duplex_config_path)

        self.listen_token_id = self.tok.convert_tokens_to_ids("<|listen|>")
        self.speak_token_id = self.tok.convert_tokens_to_ids("<|speak|>")
        self.dot_token_id = self.tok.convert_tokens_to_ids("...")
        self.tts_bos_token_id = self.tok.convert_tokens_to_ids("<|tts_bos|>")
        self.tts_eos_token_id = self.tok.convert_tokens_to_ids("<|tts_eos|>")

        self.chunk_eos_token_id = self.tok.convert_tokens_to_ids("<|chunk_eos|>")
        # <|chunk_tts_eos|> <|chunk_eos|>
        self.chunk_tts_eos_token_id = self.tok.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_token_id = self.tok.convert_tokens_to_ids("<|turn_eos|>")
        self.tts_pad_token_id = self.tok.convert_tokens_to_ids("<|tts_pad|>")

        self.chunk_terminator_token_ids = [self.listen_token_id, self.chunk_eos_token_id, self.chunk_tts_eos_token_id]
        self.turn_terminator_token_ids = [self.turn_eos_token_id]
        self.chunk_speak_token_ids = [self.speak_token_id]

    def _audio_mode(self):
        self.loop = [
            "<unit>",
            "<|audio|>", "<|audio|>", "<|audio|>", "<|audio|>", "<|audio|>",
            "<|audio|>", "<|audio|>", "<|audio|>", "<|audio|>", "<|audio|>:decode",
            "</unit>"
        ]

    def _video_mode(self):
        # 根据 FPS 生成 image tokens
        # fps=1: 1个image, fps=2: 2个image, 以此类推
        image_tokens = []
        for i in range(self.fps):
            if i == self.fps - 1:
                # 最后一个 image 带 :decode
                image_tokens.extend(["<image>", "<unk>", "</image>:decode"])
            else:
                image_tokens.extend(["<image>", "<unk>", "</image>"])
        
        self.loop = ["<unit>"] + image_tokens + ["</unit>"]

    def _omni_mode(self):
        # 根据 FPS 生成 image tokens
        # fps=1: 1个image, fps=2: 2个image, 以此类推
        image_tokens = []
        for i in range(self.fps):
            image_tokens.extend(["<image>", "<unk>", "</image>"])
        
        # 10个 audio tokens，最后一个带 :decode
        audio_tokens = ["<|audio|>"] * 9 + ["<|audio|>:decode"]
        
        self.loop = ["<unit>"] + image_tokens + audio_tokens + ["</unit>"]

class OmniDuplex(Duplex):
    def __init__(self, model, tokenizer, device='cuda', processor=None, ls_mode='implicit', 
                 duplex_config_path: Optional[str] = None):
        super().__init__(model, tokenizer, device, processor, ls_mode, duplex_config_path)

    def _prepare(self, prefix_system_prompt = None, suffix_system_prompt = None, ref_audio = None, mode = "omni", fps = None, nm_tuple = None):
        self.decoder.reset()

        # 如果传入了 fps 参数，更新 self.fps
        if fps is not None:
            self.fps = fps
        
        # 如果传入了 nm_tuple 参数，更新 stack 配置
        # nm_tuple = (n, m)，real_fps = 非零元素的个数
        if nm_tuple is not None:
            self.nm_tuple = tuple(nm_tuple)
            # real_fps = 非零 segment 的个数（即每秒实际输出的帧数）
            self.real_fps = sum(1 for x in self.nm_tuple if x != 0)
            # 使用 real_fps 作为 loop 的 fps
            self.fps = self.real_fps

        if mode == "omni":
            self._omni_mode()
        elif mode == "video":
            self._video_mode()
        elif mode == "audio":
            self._audio_mode()
        
        # 保存完整的 system prompt
        self.sys = (prefix_system_prompt or '') + (suffix_system_prompt or '')
        
        # 系统提示
        if prefix_system_prompt:
            for token_id in self.tok.encode(prefix_system_prompt, add_special_tokens=False):
                self.decoder.feed(self.decoder.embed_token(token_id))

        # 参考音频
        # 目前还没有使用过这个功能
        if ref_audio:
            data = self.processor.process_audio([ref_audio])
            embeds_nested = self.m.get_audio_embedding(data, chunk_length=self.m.config.audio_chunk_length)
            embeds = torch.cat([t for g in embeds_nested for t in g], dim=0) if embeds_nested else None
            self.decoder.feed(embeds)

        if suffix_system_prompt:
            for token_id in self.tok.encode(suffix_system_prompt, add_special_tokens=False):
                self.decoder.feed(self.decoder.embed_token(token_id))

    def generate(self, paths, items, **kwargs):
        def video_to_frames(video_path: str, frame_save_path: str, fps: int = 1, start_time: Optional[str] = None, end_time: Optional[str] = None, nm_tuple: tuple = None, sampling_mode: str = "fixed") -> str:
            """
            将视频转换为帧图像，支持 stack 模式。
            
            Args:
                video_path: 视频文件路径
                frame_save_path: 帧保存路径
                fps: 帧率（用于 ffmpeg 抽帧，stack 模式下为输入帧率如 10/20）
                start_time: 开始时间
                end_time: 结束时间
                nm_tuple: Stack 配置，如 (1, 4) 表示第一帧单独 + 第二帧是4帧stack
                         如果为 None 则使用普通帧提取模式
                sampling_mode: 抽帧模式，"uniform" (均匀抽帧) 或 "fixed" (等分定格抽帧)
                
            Returns:
                实际使用的帧保存路径（可能是原路径或 /tmp 路径）
            """
            # 检查帧是否已存在
            if os.path.exists(frame_save_path):
                existing_frames = [f for f in os.listdir(frame_save_path) if f.startswith('frame_') and f.endswith('.jpg')]
                if len(existing_frames) > 0:
                    print(f"✅ 帧已存在 ({len(existing_frames)} 帧): {frame_save_path}")
                    return frame_save_path
            
            # 尝试创建目录，如果失败则使用 /tmp
            actual_save_path = frame_save_path
            try:
                os.makedirs(frame_save_path, exist_ok=True)
                # 测试是否可写
                test_file = os.path.join(frame_save_path, '.write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (OSError, IOError) as e:
                # 目录只读，使用 /tmp
                import hashlib
                # 生成唯一的 tmp 路径
                path_hash = hashlib.md5(frame_save_path.encode()).hexdigest()[:8]
                base_name = os.path.basename(frame_save_path)
                actual_save_path = os.path.join('/tmp', 'video_frames', f'{base_name}_{path_hash}')
                
                # 检查 /tmp 中是否已有帧
                if os.path.exists(actual_save_path):
                    existing_frames = [f for f in os.listdir(actual_save_path) if f.startswith('frame_') and f.endswith('.jpg')]
                    if len(existing_frames) > 0:
                        print(f"✅ 帧已存在于 /tmp ({len(existing_frames)} 帧): {actual_save_path}")
                        return actual_save_path
                
                print(f"⚠️ 原目录只读，使用 /tmp: {actual_save_path}")
                os.makedirs(actual_save_path, exist_ok=True)
            
            # Stack 模式：使用 load_video_and_stack_frames 进行帧处理
            if nm_tuple is not None and load_video_and_stack_frames is not None:
                target_fps_from_nm = sum(nm_tuple)
                print(f"📦 Stack 模式: nm_tuple={nm_tuple}, input_fps={fps}, target_fps={target_fps_from_nm}, sampling={sampling_mode}")
                try:
                    stacked_frames, stack_info = load_video_and_stack_frames(
                        video_path,
                        start_sec=float(start_time) if start_time is not None else None,
                        end_sec=float(end_time) if end_time is not None else None,
                        fps=float(fps),
                        nm_tuple=nm_tuple,
                        sampling_mode=sampling_mode,
                    )
                    print(f"   Stack 结果: {len(stacked_frames)} 帧, real_fps={stack_info.real_fps}, target_fps={stack_info.target_fps}")
                    
                    # 保存 stack 后的帧
                    for i, frame in enumerate(stacked_frames):
                        frame_path = os.path.join(actual_save_path, f"frame_{i+1:06d}.jpg")
                        frame.save(frame_path, "JPEG", quality=95)
                    
                    print(f"✅ Stack 帧已保存: {actual_save_path}")
                    return actual_save_path
                except Exception as e:
                    print(f"⚠️ Stack 模式失败，回退到普通模式: {e}")
            
            # 普通模式：执行 ffmpeg 提取帧
            cmd = ["ffmpeg", "-y"]
            if start_time is not None:
                cmd += ["-ss", str(start_time)]
            if end_time is not None:
                cmd += ["-to", str(end_time)]
            cmd += ["-i", video_path, "-vf", f"fps={fps}", os.path.join(actual_save_path, "frame_%06d.jpg")]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            
            return actual_save_path
        assert len(paths) == 1, "Only one path is supported"
        
        dataset_name = items[0].get('dataset_name')
        audio_path = None
        frame_path = None
        
        # Get configuration from duplex_configs
        config = self.duplex_configs.get(dataset_name)
        if not config:
            raise ValueError(f"Config '{dataset_name}' not found in duplex configs")
        
        mode = config['mode']
        prefix_prompt = config['prefix_system_prompt']
        suffix_prompt = config.get('suffix_system_prompt', '')
        pad_audio = config.get('pad_audio', False)
        pad_frame = config.get('pad_frame', False)
        
        # Format system prompt
        sys_prompt = f'''<|im_start|>system\n{prefix_prompt}'''
        
        # Handle special formatting for livesports3k_cc dataset
        if 'livesports3k_cc' in dataset_name and '{title}' in prefix_prompt:
            title = items[0].get('event_title', '')
            previous = items[0].get('preasr_text', '')
            sys_prompt = sys_prompt.format(title=title, previous=previous)
        
        # Handle SWW dataset: use question as system prompt
        if config.get('use_question_as_prompt', False) and '{question}' in prefix_prompt:
            question = items[0].get('question', '')
            sys_prompt = sys_prompt.format(question=question)
        
        sys_prompt += '<|im_end|>'
        
        # 帧率配置（需要在 _prepare 之前获取，用于设置 loop 的 fps）
        frame_fps = config.get('frame_fps', 1)
        
        # Stack 配置：nm_tuple = (n, m, ...)，用于 stack 多帧成一张图
        # 如果配置了 nm_tuple，则使用 stack 模式；否则使用普通帧提取
        nm_tuple = config.get('nm_tuple', None)
        if nm_tuple is not None:
            nm_tuple = tuple(nm_tuple)  # 确保是 tuple
        
        # 抽帧模式："uniform" (均匀抽帧) 或 "fixed" (等分定格抽帧)
        sampling_mode = config.get('sampling_mode', 'fixed')
        
        # Prepare based on mode，传入 fps 和 nm_tuple 配置
        self._prepare(prefix_system_prompt=sys_prompt, suffix_system_prompt=suffix_prompt, ref_audio=None, mode=mode, fps=frame_fps, nm_tuple=nm_tuple)
        
        # Set paths based on mode
        # 获取时间切割参数（如果数据集有提供）
        start_time = items[0].get('begin', None) if items and len(items) > 0 else None
        end_time = items[0].get('end', None) if items and len(items) > 0 else None
        
        # 为有时间段的数据生成唯一的帧路径，包含 fps/nm_tuple/start/end 确保唯一性
        nm_str = f"_{nm_tuple[0]}_{nm_tuple[1]}" if nm_tuple else ""
        if start_time is not None and end_time is not None:
            frame_suffix = f'_frames_{frame_fps}fps{nm_str}_{start_time:.2f}_{end_time:.2f}'
        else:
            frame_suffix = f'_frames_{frame_fps}fps{nm_str}'
        
        if mode == 'audio':
            audio_path = paths[0].get('audio_path')
        elif mode == 'video':
            video_path = paths[0].get('video_path')
            expected_frame_path = video_path.replace('.mp4', frame_suffix)
            frame_path = video_to_frames(video_path, expected_frame_path, fps=frame_fps, start_time=start_time, end_time=end_time, nm_tuple=nm_tuple, sampling_mode=sampling_mode)
        elif mode == 'omni':
            # For omni mode, we need both audio and video
            if 'video_path' in paths[0]:
                video_path = paths[0].get('video_path')
                expected_frame_path = video_path.replace('.mp4', frame_suffix)
                frame_path = video_to_frames(video_path, expected_frame_path, fps=frame_fps, start_time=start_time, end_time=end_time, nm_tuple=nm_tuple, sampling_mode=sampling_mode)
                
                # 自动查找同名的音频文件（和 _load_waveform_and_duration 逻辑一致）
                base_path = os.path.splitext(video_path)[0]
                audio_path = None
                for ext in [".wav", ".mp3", ".m4a", ".flac"]:
                    candidate = base_path + ext
                    if os.path.exists(candidate):
                        audio_path = candidate
                        print(f"Found audio file: {audio_path}")
                        break
                
                if audio_path is None:
                    # 如果没找到同名音频，尝试使用提供的 audio_path
                    if 'audio_path' in paths[0]:
                        audio_path = paths[0].get('audio_path')
                        print(f"Using provided audio path: {audio_path}")
                    else:
                        print("Warning: No audio file found for omni mode")

        res = self.offline_generate(audio_path=audio_path, frame_path=frame_path)
        raw_sequence = self.tok.decode(res['total_ids'])
        
        # 完整 sequence：system prompt + 生成内容
        full_sequence = self.sys + raw_sequence
        
        # 从 sequence 解析出 prediction_ctc_data
        prediction_ctc_data = self._parse_sequence_to_ctc(raw_sequence)
        
        result = {
            "response": self.tok.decode(res['res_ids']),
            "sequence": full_sequence,
            "system_prompt": self.sys,  # 记录真实的 system prompt
            "prediction_ctc_data": prediction_ctc_data,
        }
        return [result]
    
    def _parse_sequence_to_ctc(self, sequence: str) -> list:
        """
        从 sequence 解析出 CTC 格式的数据
        
        规则：
        - 每个 <unit>...</unit> 块代表 1 秒
        - <|speak|> 和 <|chunk_eos|>/<|turn_eos|> 之间的内容是 LLM 说的话
        - 每次遇到标点符号就另起一个 sentence
        
        Returns:
            list: [{"sentence": "...", "start": 0.0, "end": 1.0}, ...]
        """
        import re
        
        # 分割 unit 块
        unit_pattern = r'<unit>(.*?)</unit>'
        units = re.findall(unit_pattern, sequence, re.DOTALL)
        
        # 提取每个 unit 中的说话内容
        speak_pattern = r'<\|speak\|>(.*?)(?:<\|chunk_eos\|>|<\|turn_eos\|>|<\|listen\|>|$)'
        
        # 收集所有说话片段及其时间
        speech_segments = []  # [(text, start_sec), ...]
        
        for unit_idx, unit_content in enumerate(units):
            matches = re.findall(speak_pattern, unit_content, re.DOTALL)
            for match in matches:
                text = match.strip()
                if text:
                    speech_segments.append((text, unit_idx))
        
        if not speech_segments:
            return []
        
        # 合并连续的文本，然后按标点分割成句子
        # 标点符号：。！？，；：.!?,;:
        punctuation_pattern = r'([。！？.!?])'
        
        ctc_data = []
        current_sentence = ""
        current_start = None
        
        for text, sec in speech_segments:
            if current_start is None:
                current_start = sec
            
            # 检查文本中是否有标点符号
            parts = re.split(punctuation_pattern, text)
            
            for i, part in enumerate(parts):
                if not part:
                    continue
                
                # 如果开始新句子，设置开始时间
                if current_start is None:
                    current_start = sec
                    
                current_sentence += part
                
                # 如果这个 part 是标点符号，结束当前句子
                if re.match(punctuation_pattern, part):
                    if current_sentence.strip():
                        ctc_data.append({
                            "sentence": current_sentence.strip(),
                            "start": float(current_start),
                            "end": float(sec + 1)  # unit 结束时间
                        })
                    current_sentence = ""
                    current_start = None
        
        # 处理最后一个未结束的句子
        if current_sentence.strip() and current_start is not None:
            last_sec = speech_segments[-1][1] if speech_segments else 0
            ctc_data.append({
                "sentence": current_sentence.strip(),
                "start": float(current_start),
                "end": float(last_sec + 1)
            })
        
        return ctc_data

    @torch.no_grad()
    def offline_generate(self, frame_path, audio_path, text_repetition_penalty=1.05, temperature=0.7, top_k=20, top_p=0.8, text_repetition_window_size=512, listen_prob_scale=1.0, special_text_repetition_window_size=32):
        """
        离线模式：一次性加载指定目录下的所有帧与整段音频，预计算 embeddings，
        按 LoopPlanner 计划顺序送入解码器，周期性在 audio decode 处解码文本。
        """
        res_ids = []
        MAX_NEW_SPEAK_TOKENS = 600
        MAX_NEW_SPEAK_TOKENS_PER_CHUNK = 20
        MAX_LOOP_TIME = 300 # 150s
        DECODE_MODE = "sampling"
        DEBUG_PRINT_TOP5 = False
        total_ids = []
        total_hidden = []

        iter_data = MockStreamProvider(frame_path, audio_path, self.processor) 
        audio_data, frame_data = iter_data.all_data(device=self.dev)
        
        # Handle potentially None data gracefully
        frame_emb_list = None
        if frame_data is not None:
            frame_emb_list = self.m.get_vision_embedding(frame_data)
            if frame_emb_list and len(frame_emb_list) > 0:
                frame_emb_list = frame_emb_list[0]
            else:
                frame_emb_list = None
        
        audio_emb_list = None
        if audio_data is not None:
            audio_emb_list = self.m.get_audio_embedding(audio_data, chunk_length=self.m.config.audio_chunk_length)
            if audio_emb_list and len(audio_emb_list) > 0:
                audio_emb_list = torch.cat(audio_emb_list[0], dim=0)
            else:
                audio_emb_list = None

        # 3) 循环推进
        loop = LoopPlanner(self.step_map, self.loop, self.tok)
        provider = StreamProvider(audio_emb_list, frame_emb_list, audio_chunk_len=1, image_step=1, audio_chunk_ms=100)
        speak_count = 0
        total_res_with_time = []
        # Check if we have at least one valid stream
        if audio_emb_list is None and frame_emb_list is None:
            print("Warning: No audio or frame data available, returning empty result")
            return {"total_ids": [], "res_ids": []}
        
        while (speak_count < MAX_NEW_SPEAK_TOKENS 
                and not provider.finished() 
                and provider.get_current_time() < MAX_LOOP_TIME):
            
            modal, tid, action = loop.next_plan()
            total_ids.append(tid)
            if action == 'sliding':
                self.decoder.sliding_embeds()
                continue
            embeds = None
            if modal == "audio":
                embeds = provider.next_audio()
            elif modal == "image":
                embeds = provider.next_image()
                if embeds is not None:
                    embeds = embeds.squeeze(0)
            elif modal == "sp_text":
                embeds = self.decoder.embed_token(tid)
            else:
                raise ValueError(f"Unknown modal: {modal}")

            if embeds is None:
                print(f"No embeds for modal: {modal}")
                break

            if action == "prefill":
                self.decoder.feed(embeds)
            
            elif action == "decode":
                total_hidden_in_unit = []
                logits, _ = self.decoder.feed(embeds, return_logits=True)
                current_time = provider.get_current_time()
                end_of_turn = False
                for j in range(MAX_NEW_SPEAK_TOKENS_PER_CHUNK):
                    if j == MAX_NEW_SPEAK_TOKENS_PER_CHUNK - 1:
                        if self.ls_mode == 'explicit':
                            self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))
                            total_ids.append(self.chunk_eos_token_id)
                            break
                    DEBUG_PRINT_TOP5 = False
                    last_id = self.decoder.decode(
                        logits=logits, mode=DECODE_MODE,debug_print_top5=DEBUG_PRINT_TOP5, text_repetition_penalty=text_repetition_penalty,
                        temperature=temperature, top_k=top_k, top_p=top_p, text_repetition_window_size=text_repetition_window_size, listen_prob_scale=listen_prob_scale,
                        )
                    
                    is_listen = last_id.item() == self.listen_token_id
                    if last_id.item() in self.turn_terminator_token_ids:
                        end_of_turn = True

                    
                    total_res_with_time.append(
                        {
                            "text": self.tok.decode([last_id.item()]),
                            "timestamp": provider.get_current_time()
                        }
                    )

                    print(f"\033[92m[t={current_time}] Speak: {self.tok.decode([last_id.item()])}\033[0m")
                    
                    total_ids.append(last_id.item())
                    
                    if last_id.item() in self.chunk_terminator_token_ids:
                        if self.ls_mode == 'explicit':
                            self.decoder.feed(self.decoder.embed_token(last_id.item()))
                        break
                    
                    else:
                        if last_id.item() in self.chunk_speak_token_ids or last_id.item() == self.tts_pad_token_id:
                            pass
                        else:
                            res_ids.append(last_id.item())
                            speak_count += 1
                        
                        
                        logits, hidden = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)
                        assert len(hidden.shape) == 3
                        assert hidden.shape[0] == 1
                        assert hidden.shape[1] == 1

                        
                        if j != 0:
                            total_hidden_in_unit.append([last_id.item(), hidden, end_of_turn])
                
                if is_listen:
                    pass
                else:
                    total_hidden.append(total_hidden_in_unit)
        item = {
            "total_ids": total_ids,
            "res_ids": res_ids,
            "total_res_with_time": total_res_with_time,
            "total_hidden": total_hidden
        }
        
        
        return item
