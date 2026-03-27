import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig
import json
import random
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image
import os
import re

from o_e_Kit.utils.config_utils import load_config
from o_e_Kit.utils.utils import (
    load_audio,
    load_video,
    load_video_and_audio,
    load_video_and_audio_interleaved,
)


class MiniCPM_o:
    """
    MiniCPM-O 模型基类
    
    提供通用的模型加载、配置和消息构建功能，子类可继承并实现特定的推理逻辑。
    """
    
    # 默认配置文件路径（子类可覆盖）
    DEFAULT_CONFIG_PATH: Optional[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "../../configs/generation_configs.json"
    )

    def __init__(self, model_path: str, ckpt_path: Optional[str] = None, device=None, 
                 config_path: Optional[str] = None, 
                 dataset_generation_config_path: Optional[str] = None) -> None:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        
        # 加载数据集生成配置
        config_file = dataset_generation_config_path or self.DEFAULT_CONFIG_PATH
        if config_file:
            all_configs = load_config(config_file, flatten=False)
            self.dataset_configs = load_config(config_file)
            print(f"✅ Loaded dataset generation configs from: {config_file}")
        else:
            all_configs = {}
            self.dataset_configs = {}
            print(f"⚠️ No dataset generation config loaded")
        
        # 加载 OOM 重试默认配置
        self.oom_defaults = all_configs.get("_oom_defaults", {
            "oom_strategy": "speed",
            "max_audio_speed": 5.0,
            "max_audio_trim_end": 0.8,
            "audio_speed_increment": 0.2,
            "audio_trim_increment": 0.2,
        })
        print(f"  OOM strategy: {self.oom_defaults.get('oom_strategy', 'speed')}")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # 初始化模型
        self._init_model(model_path, ckpt_path, config_path)
        
        torch.cuda.empty_cache()
        print(f"✅ 模型加载完成")
    
    def _init_model(self, model_path: str, ckpt_path: Optional[str], config_path: Optional[str]):
        """
        初始化模型、tokenizer 和 processor
        """
        print("📦 初始化模型...")
        
        # 加载配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        
        if config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(config, k, v)
                else:
                    setattr(config, key, value)
            
            print(f"  pool-step: {config.audio_pool_step}, audio_chunk_length: {config.audio_chunk_length}")
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_path, 
            config=config, 
            trust_remote_code=True, 
            local_files_only=True
        )
        
        # 加载 checkpoint（可选）
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
            
            if "module" in state_dict:
                state_dict = state_dict["module"]
            
            # 过滤 TTS 参数
            params_dict = {k: v for k, v in state_dict.items() if not k.startswith("tts")}
            
            missing_keys, unexpected_keys = self.model.load_state_dict(params_dict, strict=False)
            print(f"missing_keys: {missing_keys}\nunexpected_keys: {unexpected_keys}")
        
        self.model.eval()
        self.model.to(self.device)
        self.model = self.model.to(torch.bfloat16)
        
        # 加载 tokenizer 和 processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        
        print(f"  pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"  eos_token_id: {self.tokenizer.eos_token_id}")

    def get_generation_config(self, dataset_name: str, **kwargs) -> dict:
        """
        获取数据集的生成配置
        """
        config = self.dataset_configs.get(dataset_name)
        
        if config is None:
            raise ValueError(f"No config found for dataset '{dataset_name}'")
        
        gen_config = {
            "max_tokens": int(config.get("max_tokens", 256)),
            "num_beams": int(config.get("num_beams", 3)),
            "chunk_secs": int(config.get("chunk_secs", 30)),
            "sampling": config.get("sampling", False),
            "top_p": float(config.get("top_p", 0.8)),
            "top_k": int(config.get("top_k", 100)),
            "temperature": float(config.get("temperature", 0.7)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.02)),
            "enable_thinking": config.get("enable_thinking", False),
            "max_frames": int(config.get("max_frames", 16)),
            "max_fps": float(config.get("max_fps", 1.0)),
            "user_prompt": config.get("user_prompt", "{media}\n{question}\n{options}"),
            "system_prompt": config.get("system_prompt", ""),
            "load_av": config.get("load_av", False),
            "keep_placeholder": config.get("keep_placeholder", False),
            "interleave_fps": float(config.get("interleave_fps", 0.0)),
            "use_image_id": config.get("use_image_id", False),
            "max_slice_nums": int(config.get("max_slice_nums", 1)),
        }
        
        for key in gen_config.keys():
            if key in kwargs:
                gen_config[key] = kwargs[key]
        
        if gen_config["sampling"] and gen_config["num_beams"] > 1:
            print(f"⚠️ Warning: sampling=True 时 num_beams 必须为 1，已自动重置 num_beams=1")
            gen_config["num_beams"] = 1
        
        return gen_config
    
    def _build_options_prompt(self, choices: list) -> str:
        """构建选项提示文本"""
        if not choices:
            return ''
        
        options_prompt = ''
        KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        for key, choice in zip(KEYS[:len(choices)], choices):
            options_prompt += f'{key}. {choice}\n'
        return options_prompt
    
    # ==================== 消息构建 ====================
    
    def build_content(self, paths: Dict[str, Any], prompt: str, max_frames: int = 16, 
                       max_fps: float = 1.0, load_av: bool = False, keep_placeholder: bool = False,
                       interleave_fps: float = 0.0, audio_speed: float = 1.0, audio_trim_end: float = 0.0) -> List[Any]:
        """
        构建消息 content 列表
        
        支持的路径格式：
        - 单路径：video_path, image_path, audio_path, audio_path_list
        - dict 格式：video_paths_dict, image_paths_dict, audio_paths_dict
        """
        content = []
        
        # 检查是否有 dict 格式路径
        has_dict_paths = (paths.get('audio_paths_dict') or 
                         paths.get('image_paths_dict') or 
                         paths.get('video_paths_dict'))
        if has_dict_paths:
            content = self._build_content_with_placeholders(
                paths, prompt, max_frames, max_fps, load_av, keep_placeholder, interleave_fps, audio_speed, audio_trim_end
            )
        else:
            # 单路径格式：根据占位符位置放置媒体
            media_list = []
            
            # 加载视频
            video_path = paths.get('video_path')
            if video_path:
                if load_av:
                    if interleave_fps > 0:
                        media = load_video_and_audio_interleaved(
                            video_path,
                            max_frames=max_frames,
                            max_fps=max_fps,
                            audio_sr=16000,
                            audio_speed=audio_speed,
                            audio_trim_end=audio_trim_end,
                        )
                        media_list.extend(media)
                    else:
                        frames, waveform = load_video_and_audio(
                            video_path,
                            max_frames=max_frames,
                            audio_sr=16000,
                            max_fps=max_fps,
                            audio_speed=audio_speed,
                            audio_trim_end=audio_trim_end,
                        )
                        media_list.extend(frames)
                        if waveform is not None:
                            media_list.append(waveform)
                else:
                    frames = load_video(video_path, max_frames=max_frames, max_fps=max_fps)
                    media_list.extend(frames)
            
            # 加载图片
            image_path = paths.get('image_path')
            if image_path:
                media_list.append(Image.open(image_path).convert("RGB"))
            
            # 加载音频
            audio_path = paths.get('audio_path')
            if audio_path:
                media_list.append(load_audio(audio_path, sr=16000, speed=audio_speed, trim_end=audio_trim_end))
            
            # 根据 {media} 或 {audio} 占位符位置构建 content（两者不会同时出现）
            if "{media}" in prompt or "{audio}" in prompt:
                placeholder = "{media}" if "{media}" in prompt else "{audio}"
                parts = prompt.split(placeholder)
                for i, part in enumerate(parts):
                    if part.strip():
                        content.append(part.strip())
                    if i == 0 and media_list:
                        content.extend(media_list)
            else:
                # 没有 {media} 占位符，媒体放前面
                content.extend(media_list)
                if prompt.strip():
                    content.append(prompt.strip())
        
        return content
    
    def _build_content_with_placeholders(self, paths: Dict[str, Any], prompt: str, max_frames: int,
                                          max_fps: float = 1.0, load_av: bool = False, keep_placeholder: bool = False,
                                          interleave_fps: float = 0.0, audio_speed: float = 1.0, audio_trim_end: float = 0.0) -> List[Any]:
        """处理 dict 格式路径，在 prompt 中处理占位符和媒体内容"""
        placeholders = {}
        
        for key, audio_path in paths.get('audio_paths_dict', {}).items():
            if audio_path:
                placeholders[key] = load_audio(audio_path, sr=16000, speed=audio_speed, trim_end=audio_trim_end)
        
        for key, image_path in paths.get('image_paths_dict', {}).items():
            if image_path:
                placeholders[key] = Image.open(image_path).convert("RGB")
        
        for key, video_path in paths.get('video_paths_dict', {}).items():
            if video_path:
                if load_av:
                    if interleave_fps > 0:
                        media = load_video_and_audio_interleaved(
                            video_path,
                            max_frames=max_frames,
                            max_fps=max_fps,
                            audio_sr=16000,
                            audio_speed=audio_speed,
                            audio_trim_end=audio_trim_end,
                        )
                    else:
                        frames, waveform = load_video_and_audio(
                            video_path,
                            max_frames=max_frames,
                            audio_sr=16000,
                            max_fps=max_fps,
                            audio_speed=audio_speed,
                            audio_trim_end=audio_trim_end,
                        )
                        media = frames + ([waveform] if waveform is not None else [])
                    placeholders[key] = media
                else:
                    placeholders[key] = load_video(video_path, max_frames=max_frames, max_fps=max_fps)
        
        prompt = prompt.replace("{media}", "").strip()
        
        if not placeholders:
            return [prompt]
        
        content = []
        pattern = '(' + '|'.join(re.escape(k) for k in placeholders.keys()) + ')'
        parts = re.split(pattern, prompt)
        
        for part in parts:
            if not part:
                continue
            if part in placeholders:
                if keep_placeholder:
                    content.append(part)
                media = placeholders[part]
                if isinstance(media, list):
                    content.extend(media)
                else:
                    content.append(media)
            else:
                if part:
                    content.append(part)
        
        return content
    
    def build_messages(self, dataset_name: str, paths: Dict[str, Any], item: Dict[str, Any], 
                        audio_speed: float = 1.0, audio_trim_end: float = 0.0) -> tuple:
        """
        构建完整的消息列表
        
        Returns:
            (msgs, gen_config) 元组
        """
        gen_config = self.get_generation_config(dataset_name)
        max_frames = gen_config["max_frames"]
        max_fps = gen_config["max_fps"]
        user_prompt_template = gen_config["user_prompt"]
        system_prompt = gen_config["system_prompt"]
        load_av = gen_config["load_av"]
        keep_placeholder = gen_config["keep_placeholder"]
        interleave_fps = gen_config["interleave_fps"]
        
        question = item.get('question', item.get('prompt', ''))
        choices = item.get('choices', [])
        options_prompt = self._build_options_prompt(choices)
        sqa_context = item.get('sqa_context', '')
        
        prompt = user_prompt_template.replace("{question}", question).replace("{options}", options_prompt.rstrip())
        prompt = prompt.replace("{sqa_context}", sqa_context)
        
        content = self.build_content(paths, prompt, max_frames, max_fps, load_av, keep_placeholder, interleave_fps, audio_speed, audio_trim_end)
        
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": [system_prompt]})
        msgs.append({"role": "user", "content": content})
        
        return msgs, gen_config
    
    def _msgs_to_str(self, msgs: List[Dict]) -> str:
        """将 msgs 转为字符串，非文本内容用占位符替换"""
        result = []
        for msg in msgs:
            role = msg.get('role', '')
            content = msg.get('content', [])
            
            content_strs = []
            audio_idx = 0
            image_idx = 0
            
            for c in content:
                if isinstance(c, str):
                    content_strs.append(c)
                elif isinstance(c, np.ndarray):
                    audio_idx += 1
                    content_strs.append(f"<audio_{audio_idx}>")
                elif isinstance(c, Image.Image):
                    image_idx += 1
                    content_strs.append(f"<image_{image_idx}>")
                else:
                    content_strs.append("<unknown>")
            
            result.append(f"[{role}]: {' '.join(content_strs)}")
        
        return "\n".join(result)
    
    # ==================== 推理接口 ====================
    
    def generate_chat(self, paths: list[dict], items: list[dict], dataset_name: str, **kwargs) -> list[Dict[str, Any]]:
        """
        Chat 模式推理接口（批量处理）
        
        Args:
            paths: 包含媒体路径的字典列表
            items: 包含 prompt 和 choices 等信息的字典列表
            dataset_name: 数据集名称
        
        Returns:
            包含 response 和 sequence 的字典列表
        """
        results = []
        for path, item in zip(paths, items):
            result = self._generate_chat(path, item, dataset_name, **kwargs)
            results.append(result)
        return results
    
    def _generate_chat(self, paths: dict, items: dict, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """
        使用 model.chat() API 进行推理，支持 OOM 自动重试
        """
        # OOM 重试参数（从配置读取）
        oom_strategy = self.oom_defaults.get("oom_strategy", "speed")
        max_audio_speed = self.oom_defaults.get("max_audio_speed", 5.0)
        max_trim_end = self.oom_defaults.get("max_audio_trim_end", 0.8)
        speed_increment = self.oom_defaults.get("audio_speed_increment", 0.2)
        trim_increment = self.oom_defaults.get("audio_trim_increment", 0.2)
        
        audio_speed = 1.0
        audio_trim_end = 0.0
        
        while True:
            # 构建消息（传入 audio_speed 和 audio_trim_end 用于 OOM 重试）
            msgs, gen_config = self.build_messages(dataset_name, paths, items,
                                                   audio_speed=audio_speed, audio_trim_end=audio_trim_end)
            max_tokens = gen_config["max_tokens"]
            use_image_id = gen_config.get("use_image_id", False)
            max_slice_nums = gen_config.get("max_slice_nums", 1)
            interleave_fps = gen_config.get("interleave_fps", 0.0)
            
            # 分析 content 中的模态
            content = msgs[-1]["content"]
            has_audio = any(isinstance(c, np.ndarray) for c in content)
            has_vision = any(isinstance(c, Image.Image) for c in content)
            omni_input = has_audio and has_vision and interleave_fps > 0
            
            # 根据路径类型调整视觉参数
            has_video_path = bool(paths.get("video_path")) or bool(paths.get("video_paths_dict"))
            has_image_path = bool(paths.get("image_path")) or bool(paths.get("image_paths_dict"))
            
            merge_audio_from_same_content = True
            if (not has_video_path) and has_image_path:
                max_slice_nums = 9
                use_image_id = True
                merge_audio_from_same_content = False
            else:
                max_slice_nums = 1
                use_image_id = False
                merge_audio_from_same_content = True
            
            if audio_speed > 1.0 or audio_trim_end > 0:
                audio_count = sum(1 for c in content if isinstance(c, np.ndarray))
                info = "⚡ OOM重试: "
                if audio_trim_end > 0:
                    info += f"trim_end={audio_trim_end:.1f}s "
                if audio_speed > 1.0:
                    info += f"speed={audio_speed:.1f}x "
                info += f"({audio_count} 段音频)"
                print(info)
            
            try:
                # 构建 chat 参数（与 baseline minicpmo_ou.py 对齐）
                response = self.model.chat(
                    msgs=msgs,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                    max_inp_length=32768,  # 支持长视频处理
                    use_tts_template=True,
                    use_image_id=use_image_id,
                    max_slice_nums=max_slice_nums,
                    omni_mode=omni_input,
                    merge_audio_from_same_content=merge_audio_from_same_content,
                )
                
                if isinstance(response, list):
                    response = response[0]
                if isinstance(response, tuple):
                    response = response[0]
                if isinstance(response, str):
                    response = response.replace("<|tts_eos|>", "").strip()
                
                return {
                    "response": response,
                    "sequence": self._msgs_to_str(msgs),
                    "audio_speed": audio_speed,
                    "audio_trim_end": audio_trim_end,
                }
            
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # 检查是否是 OOM
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                if not is_oom:
                    print(f"❌ 推理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return {"response": "", "sequence": self._msgs_to_str(msgs), "error": str(e)}
                
                # OOM：清理显存
                torch.cuda.empty_cache()
                
                # 根据配置的策略进行重试
                if oom_strategy == "trim":
                    audio_trim_end += trim_increment
                    print(f"⚠️ OOM detected, will retry with audio_trim_end={audio_trim_end:.1f}s")
                elif oom_strategy == "speed":
                    if audio_speed < max_audio_speed:
                        audio_speed += speed_increment
                        print(f"⚠️ OOM detected, will retry with audio_speed={audio_speed:.1f}x")
                    else:
                        print(f"❌ OOM 无法恢复: 已尝试 speed={max_audio_speed}x")
                        return {"response": "", "sequence": "", "error": f"OOM after max speed={max_audio_speed}x"}
                elif oom_strategy == "speed_then_trim":
                    if audio_speed < max_audio_speed:
                        audio_speed += speed_increment
                        if audio_speed > max_audio_speed:
                            audio_speed = max_audio_speed
                        print(f"⚠️ OOM detected, will retry with audio_speed={audio_speed:.1f}x")
                    else:
                        audio_trim_end += trim_increment
                        print(f"⚠️ OOM detected, will retry with audio_trim_end={audio_trim_end:.1f}s (speed={audio_speed:.1f}x)")
                else:
                    print(f"❌ 未知的 OOM 策略: {oom_strategy}")
                    return {"response": "", "sequence": "", "error": f"Unknown oom_strategy: {oom_strategy}"}
                continue
            
            except Exception as e:
                print(f"❌ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "response": "",
                    "sequence": self._msgs_to_str(msgs),
                    "error": str(e),
                }
