import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from o_e_Kit.utils.config_utils import load_config

try:
    # 官方多模态工具包：pip install qwen-omni-utils -U
    from qwen_omni_utils import process_mm_info
except ImportError as e:  # pragma: no cover - 运行环境缺少依赖时给出清晰提示
    raise ImportError(
        "无法导入 'qwen_omni_utils'。请先安装依赖：\n"
        "  pip install qwen-omni-utils -U"
    ) from e


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../configs/generation_configs.json",
)


class Qwen3OmniEvalModel:
    """
    Qwen3-Omni 评测封装

    - 内容构建逻辑严格按照官方示例：
      conversation -> apply_chat_template -> process_mm_info -> processor(...) -> model.generate(...)
    - 上层评测逻辑与 MiniCPM-O-OU 保持接口兼容：
      通过 infer.run_model_generation(..., generate_method=\"generate\") 调用本类的 generate(...)
    """

    def __init__(
        self,
        model_path: str,
        ckpt_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        config_path: Optional[str] = None,
        dataset_generation_config_path: Optional[str] = None,
    ) -> None:
        # 随机种子
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # 加载数据集生成配置（与 MiniCPM-O-OU 复用同一份 omni_generation_configs.json）
        if dataset_generation_config_path:
            self.dataset_configs = load_config(dataset_generation_config_path)
            print(
                f"✅ Loaded omni dataset generation configs from: "
                f"{dataset_generation_config_path}"
            )
        else:
            self.dataset_configs = load_config(DEFAULT_CONFIG_PATH)
            print(
                f"✅ Using default omni dataset generation configs: "
                f"{DEFAULT_CONFIG_PATH}"
            )

        # 对于 Qwen3-Omni，我们优先依赖 HuggingFace 的 device_map 自动分片
        # 这里的 device 主要用于日志和兼容接口，不再用于 model.to(device)
        self.device: torch.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path

        self._init_model_and_processor(model_path, config_path)

        # 目前评测只用文本输出，音频输出如有需要可以后续扩展
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # 模型 / Processor 初始化
    # ----------------------------------------------------------------------
    def _init_model_and_processor(
        self, model_path: str, config_path: Optional[str]
    ) -> None:
        """
        按官方示例方式初始化 Qwen3-Omni：

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype=\"auto\",
            attn_implementation=\"flash_attention_2\",
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        """
        print("📦 初始化 Qwen3-Omni 模型...")

        # Qwen3-Omni 自身带有完整 config，这里不额外叠加 config_path
        # 使用 device_map=\"auto\" 让 HF 根据 CUDA_VISIBLE_DEVICES 自动做多卡分片
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()

        self.model.eval()

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

        # 记录模型主设备（在多卡分片时通常是第一块 GPU）
        try:
            model_device = getattr(self.model, "device", None)
        except Exception:
            model_device = None

        if model_device is not None:
            self.device = model_device

        print(f"  device: {self.device}")
        try:
            print(f"  model dtype: {self.model.dtype}")
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # 与 MiniCPM-O-OU 一致的配置（只复用文本相关配置）
    # ----------------------------------------------------------------------
    def get_generation_config(self, dataset_name: str) -> Dict[str, Any]:
        """获取某个数据集的生成配置（从 omni_generation_configs.json 展平后字典中读取）"""
        config = self.dataset_configs.get(dataset_name)
        if config is None:
            raise ValueError(f"No omni generation config found for dataset '{dataset_name}'")

        return {
            "max_tokens": int(config.get("max_tokens", 256)),
            "user_prompt": config.get(
                "user_prompt", "{media}\n{question}\n{options}"
            ),
            "system_prompt": config.get("system_prompt", ""),
            "load_av": bool(config.get("load_av", False)),
        }

    def _build_options_prompt(self, choices: List[str]) -> str:
        """
        将选项列表格式化为：
        \"A. xxx\\nB. yyy\\n\"
        """
        if not choices:
            return ""

        options_prompt = ""
        keys = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        for key, choice in zip(keys[: len(choices)], choices):
            options_prompt += f"{key}. {choice}\n"
        return options_prompt

    # ----------------------------------------------------------------------
    # conversation 构建（官方 Qwen3-Omni 示例格式）
    # ----------------------------------------------------------------------
    def build_conversation(
        self, dataset_name: str, paths: Dict[str, Any], item: Dict[str, Any]
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        """
        将 omni 数据集样本转换为 Qwen3-Omni 官方示例的 conversation 结构：

        conversation = [
            {
                \"role\": \"user\",
                \"content\": [
                    {\"type\": \"image\", \"image\": ...},
                    {\"type\": \"audio\", \"audio\": ...},
                    {\"type\": \"text\",  \"text\":  \"...\"}
                ],
            },
        ]
        """
        gen_config = self.get_generation_config(dataset_name)
        user_prompt_template = gen_config["user_prompt"]
        system_prompt = gen_config["system_prompt"]

        # 文本部分（与 MiniCPM-O-OU 保持一致）
        question = item.get("question", item.get("prompt", ""))
        choices = item.get("choices", [])
        options_prompt = self._build_options_prompt(choices)
        sqa_context = item.get("sqa_context", "")

        # 文本 prompt：去掉 {media} 占位符，媒体由 conversation 的 typed content 承载
        prompt = (
            user_prompt_template.replace("{question}", question)
            .replace("{options}", options_prompt.rstrip())
            .replace("{sqa_context}", sqa_context)
            .replace("{media}", "")
            .strip()
        )

        conversation: List[Dict[str, Any]] = []

        # system prompt
        if system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content: List[Dict[str, Any]] = []

        # 单路径媒体：直接给出本地路径或 URL，交由官方 process_mm_info 处理
        video_path = paths.get("video_path")
        if video_path:
            user_content.append({"type": "video", "video": video_path})

        image_path = paths.get("image_path")
        if image_path:
            user_content.append({"type": "image", "image": image_path})

        audio_path = paths.get("audio_path")
        if audio_path:
            user_content.append({"type": "audio", "audio": audio_path})

        # dict 格式媒体（UNO-Bench / AV-Odyssey 等）
        audio_paths_dict = paths.get("audio_paths_dict") or {}
        for _, p in audio_paths_dict.items():
            if p:
                user_content.append({"type": "audio", "audio": p})

        image_paths_dict = paths.get("image_paths_dict") or {}
        for _, p in image_paths_dict.items():
            if p:
                user_content.append({"type": "image", "image": p})

        video_paths_dict = paths.get("video_paths_dict") or {}
        for _, p in video_paths_dict.items():
            if p:
                user_content.append({"type": "video", "video": p})

        # 文本部分
        if prompt:
            user_content.append({"type": "text", "text": prompt})

        conversation.append({"role": "user", "content": user_content})

        return conversation, gen_config

    # ----------------------------------------------------------------------
    # 评测用统一接口：generate（供 infer.run_model_generation 调用）
    # ----------------------------------------------------------------------
    def generate(
        self,
        dataset_name: str,
        paths: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        modality: str = "omni",
        **unused_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        统一推理接口：
        - 与 infer.run_model_generation(generate_method=\"generate\") 对齐
        - 对于每个样本构建一个 conversation，走官方 Qwen3-Omni 推理流程
        """
        results: List[Dict[str, Any]] = []

        if modality not in {"omni", "audio", "video", "image"}:
            # 对 Omni 评测来说，主要是 omni；其他模态暂时不做特殊区分
            print(f"[Qwen3-Omni] Unexpected modality: {modality}, treat as omni.")

        for path, item in zip(paths, items):
            conversation, gen_config = self.build_conversation(
                dataset_name, path, item
            )

            # Official: text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            # Official: audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            use_audio_in_video = bool(gen_config.get("load_av", False))
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=use_audio_in_video
            )

            # Official: inputs = processor(..., audio=audios, images=images, videos=videos, ...)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )

            # 按官方示例，先将 inputs 移动到模型主设备，再转换到模型 dtype
            # 在 device_map=\"auto\" 场景下，self.model.device 通常指向首个 CUDA 设备
            inputs = inputs.to(self.device)
            try:
                # 一些 BatchEncoding 实现支持 .to(dtype)，保持与官方示例一致
                inputs = inputs.to(self.model.dtype)  # type: ignore[arg-type]
            except Exception:
                pass

            # Official:
            # text_ids, audio = model.generate(..., thinker_return_dict_in_generate=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            text_ids, audio_out = self.model.generate(
                **inputs,
                speaker="Ethan",
                thinker_return_dict_in_generate=True,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=int(gen_config.get("max_tokens", 256)),
            )

            # 解码文本（与官方示例一致：跳过 prompt 部分）
            try:
                input_len = inputs["input_ids"].shape[1]
                sequences = text_ids.sequences  # type: ignore[attr-defined]
                decoded = self.processor.batch_decode(
                    sequences[:, input_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                response_text = decoded[0] if isinstance(decoded, list) else decoded
            except Exception as e:
                print(f"[Qwen3-Omni] decode failed: {e}")
                response_text = ""

            results.append(
                {
                    "response": response_text,
                    # 保留完整的生成序列信息便于调试
                    "sequence": str(conversation),
                }
            )

        return results


