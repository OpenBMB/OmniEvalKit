"""
模型加载模块
负责加载各种类型的模型
"""

import os
import torch
import json
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoProcessor


def load_model_hf(args, device):
    """加载HuggingFace格式的模型"""
    print(f"Loading tokenizer and model... model_path: {args.model_path}, tokenizer_path: {args.tokenizer_path}, config_path: {args.config_path}, pt_path: {args.pt_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|listen|>")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    setattr(config, k, v)
            else:
                setattr(config, key, value)
    print(f"pool-step: {config.audio_pool_step}")
    print(f"audio_chunk_length: {config.audio_chunk_length}")

    model = AutoModel.from_pretrained(args.model_path, config=config, trust_remote_code=True)
    model = model.to(torch.bfloat16)

    if args.pt_path:
        ckpt = torch.load(args.pt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print(f"Checkpoint loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

    model.eval().to(device)
    print("Model loaded successfully.")
    
    return tokenizer, model, processor


def load_model(args, device, duplex_type=None):
    """根据模型类型加载相应的模型"""
    if args.model_type == 'minicpmo_duplex_demo':
        args.generate_method = "generate"
        from o_e_Kit.models.minicpm.demo.duplex_runner import OmniDuplex
        tokenizer, base_model, processor = load_model_hf(args, device)
        model = OmniDuplex(model=base_model, tokenizer=tokenizer, device=device, 
                          processor=processor, ls_mode=args.ls_mode, 
                          duplex_config_path=args.dataset_generation_config_path)
    elif args.model_type == 'minicpmo':
        from o_e_Kit.models.minicpm.minicpmo import MiniCPM_o
        model = MiniCPM_o(
            args.model_path, 
            args.pt_path, 
            device, 
            args.config_path,
            dataset_generation_config_path=getattr(args, 'dataset_generation_config_path', None)
        )
    elif args.model_type == 'whisper':
        args.generate_method = "batch"
        from o_e_Kit.models.asr.whisper import Whisper
        model = Whisper(model_path=args.model_path, device=device, batch_size=args.batchsize)
    elif args.model_type == 'qwen3_omni':
        # Qwen3-Omni 多模态理解模型
        # 统一使用 generate 接口（对应 infer.run_model_generation 中的 generate_method="generate"）
        from o_e_Kit.models.qwen.qwen3_omni import Qwen3OmniEvalModel

        if not getattr(args, "generate_method", None):
            args.generate_method = "generate"
        else:
            # 非 generate 时给出提示，但仍然覆盖为 generate，避免推理阶段报错
            if args.generate_method != "generate":
                print(
                    f"⚠️ qwen3_omni 仅支持 generate 推理，"
                    f"已将 generate_method='{args.generate_method}' 覆盖为 'generate'"
                )
                args.generate_method = "generate"

        model = Qwen3OmniEvalModel(
            model_path=args.model_path,
            ckpt_path=args.pt_path,
            device=device,
            config_path=args.config_path,
            dataset_generation_config_path=getattr(
                args, "dataset_generation_config_path", None
            ),
        )
    elif args.model_type == 'gemini_omni':
        # Gemini 多模态 API 评测模型（通过 OpenAI 兼容网关调用）
        from o_e_Kit.models.gemini import GeminiOmniApiEvalModel

        if not getattr(args, "generate_method", None):
            args.generate_method = "generate"
        else:
            if args.generate_method != "generate":
                print(
                    f"⚠️ gemini_omni 仅支持 generate 推理，"
                    f"已将 generate_method='{args.generate_method}' 覆盖为 'generate'"
                )
                args.generate_method = "generate"

        api_url = os.getenv("GEMINI_API_URL", "")
        api_key = os.getenv("GEMINI_API_KEY", "")

        model = GeminiOmniApiEvalModel(
            model_name=args.model_name,
            api_url=api_url,
            api_key=api_key,
            dataset_generation_config_path=getattr(
                args, "dataset_generation_config_path", None
            ),
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    return model