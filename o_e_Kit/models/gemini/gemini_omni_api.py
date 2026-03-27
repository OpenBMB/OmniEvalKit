import base64
import mimetypes
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

from o_e_Kit.utils.config_utils import load_config


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../configs/generation_configs.json",
)


class GeminiOmniApiEvalModel:
    """
    Gemini 多模态 API 评测封装

    - 使用 OpenAI 兼容的 chat/completions 网关（通过 GEMINI_API_URL 环境变量配置）
    - 消息结构示例：
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,..." }},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..." }},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..." }}
          ]
        }

    - 接口对上层评测逻辑呈现为一个“模型”，暴露 generate(...) 方法，
      以便 infer.run_model_generation(generate_method="generate") 直接调用。
    """

    def __init__(
        self,
        model_name: str,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dataset_generation_config_path: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        random.seed(0)

        self.model_name = model_name
        self.api_url = api_url or os.getenv("GEMINI_API_URL", "")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Gemini API 评测需要提供 API Key，请设置环境变量 GEMINI_API_KEY。"
            )

        # 加载数据集生成配置（prompt / max_tokens 等）
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

        self.timeout = timeout
        self.session = requests.Session()

    # ----------------------------------------------------------------------
    # 配置与 prompt 构建
    # ----------------------------------------------------------------------
    def get_generation_config(self, dataset_name: str) -> Dict[str, Any]:
        config = self.dataset_configs.get(dataset_name)
        if config is None:
            raise ValueError(
                f"No omni generation config found for dataset '{dataset_name}'"
            )

        return {
            "max_tokens": int(config.get("max_tokens", 256)),
            "user_prompt": config.get(
                "user_prompt", "{media}\n{question}\n{options}"
            ),
            "system_prompt": config.get("system_prompt", ""),
        }

    def _build_options_prompt(self, choices: List[str]) -> str:
        if not choices:
            return ""
        keys = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        return "".join(
            f"{k}. {c}\n" for k, c in zip(keys[: len(choices)], choices)
        )

    # ----------------------------------------------------------------------
    # 媒体编码工具：本地文件 -> data:xxx;base64,....
    # ----------------------------------------------------------------------
    @staticmethod
    def _guess_mime_type(path: str, fallback: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or fallback

    @staticmethod
    def _file_to_data_url(path: str, fallback_mime: str) -> str:
        mime_type = GeminiOmniApiEvalModel._guess_mime_type(path, fallback_mime)
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    # ----------------------------------------------------------------------
    # 将 Omni 样本转换为 Gemini 网关的 messages 结构
    # ----------------------------------------------------------------------
    def build_messages(
        self, dataset_name: str, paths: Dict[str, Any], item: Dict[str, Any]
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        """
        构造 OpenAI 兼容的 messages：
        [
          {"role": "system", "content": [{"type": "text", "text": "..."}]},
          {"role": "user",   "content": [text_part, media_parts...]}
        ]
        """
        gen_config = self.get_generation_config(dataset_name)
        user_prompt_template = gen_config["user_prompt"]
        system_prompt = gen_config["system_prompt"]

        # 文本部分
        question = item.get("question", item.get("prompt", ""))
        choices = item.get("choices", [])
        options_prompt = self._build_options_prompt(choices)
        sqa_context = item.get("sqa_context", "")

        prompt = (
            user_prompt_template.replace("{question}", question)
            .replace("{options}", options_prompt.rstrip())
            .replace("{sqa_context}", sqa_context)
            .replace("{media}", "")
            .strip()
        )

        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content: List[Dict[str, Any]] = []

        # 文本部分先放进去
        if prompt:
            user_content.append({"type": "text", "text": prompt})

        # 单路径媒体：视频 / 图片 / 音频
        video_path = paths.get("video_path")
        if video_path and os.path.exists(video_path):
            url = self._file_to_data_url(video_path, "video/mp4")
            user_content.append(
                {
                    "type": "image_url",  # 网关约定：视频也使用 image_url + data:video/mp4;base64
                    "image_url": {"url": url},
                }
            )

        image_path = paths.get("image_path")
        if image_path and os.path.exists(image_path):
            url = self._file_to_data_url(image_path, "image/jpeg")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

        audio_path = paths.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            url = self._file_to_data_url(audio_path, "audio/wav")
            user_content.append(
                {
                    "type": "audio_url",  # 约定的音频类型，网关侧需支持
                    "audio_url": {"url": url},
                }
            )

        # dict 格式媒体（UNO-Bench / AV-Odyssey 等）
        for _, p in (paths.get("audio_paths_dict") or {}).items():
            if p and os.path.exists(p):
                url = self._file_to_data_url(p, "audio/wav")
                user_content.append(
                    {"type": "audio_url", "audio_url": {"url": url}}
                )

        for _, p in (paths.get("image_paths_dict") or {}).items():
            if p and os.path.exists(p):
                url = self._file_to_data_url(p, "image/jpeg")
                user_content.append(
                    {"type": "image_url", "image_url": {"url": url}}
                )

        for _, p in (paths.get("video_paths_dict") or {}).items():
            if p and os.path.exists(p):
                url = self._file_to_data_url(p, "video/mp4")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

        messages.append({"role": "user", "content": user_content})
        return messages, gen_config

    # ----------------------------------------------------------------------
    # 统一 generate 接口（供 infer.run_model_generation 调用）
    # ----------------------------------------------------------------------
    def generate(
        self,
        dataset_name: str,
        paths: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        modality: str = "omni",
        **unused_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for path, item in zip(paths, items):
            messages, gen_config = self.build_messages(dataset_name, path, item)
            max_tokens = int(gen_config.get("max_tokens", 256))

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # 简单的重试机制，避免偶发网络错误影响评测
            max_attempts = 8
            last_error: Optional[Exception] = None
            data: Dict[str, Any] = {}

            for attempt in range(1, max_attempts + 1):
                try:
                    resp = self.session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    wait = 1
                    if attempt < max_attempts:
                        print(
                            f"⚠️ Gemini API 请求失败 (第{attempt}次)，{wait}s 后重试: {e}"
                        )
                        time.sleep(wait)
                    else:
                        print(
                            f"❌ Gemini API 请求失败 (已重试 {max_attempts} 次放弃): {e}"
                        )

            sequence_text = item.get("question", item.get("prompt", ""))

            if last_error is not None:
                # 失败：记录错误信息，并在 other 中打标，后续评估时可选择跳过这些样本
                results.append(
                    {
                        "response": "",
                        "sequence": sequence_text,
                        "other": {
                            "api_ok": False,
                            "api_error": repr(last_error),
                        },
                    }
                )
                continue

            # 解析 OpenAI 兼容响应
            response_text = ""
            try:
                choices = data.get("choices") or []
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        # 兼容 content 为多段的情况，只拼接 text 段
                        texts = [
                            c.get("text", "")
                            for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        ]
                        response_text = " ".join(t for t in texts if t)
            except Exception as e:
                print(f"❌ Gemini API 响应解析失败: {e}")
                response_text = ""

            # 成功：保留响应文本和简洁的 sequence，并在 other 中标记 api_ok=True
            results.append(
                {
                    "response": response_text,
                    "sequence": sequence_text,
                    "other": {
                        "api_ok": True,
                    },
                }
            )

        return results


