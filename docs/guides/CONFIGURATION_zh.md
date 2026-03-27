# 环境变量配置

本项目使用环境变量来配置 API 密钥、模型路径等参数。你可以在项目根目录创建 `.env` 文件，程序会自动加载。

## LLM 评估（LLM Fallback 评估时必需）

```bash
# OpenAI 兼容 API
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions

# 支持多 key 轮换，按序号配置
OPENAI_API_KEY_1=sk-xxx
OPENAI_API_KEY_2=sk-yyy
```

当规则匹配和 Sentence Transformer 语义匹配均无法判定时，系统会自动调用 LLM API 进行兜底评估。

## Gemini API（可选，仅使用 gemini_omni 模型时需要）

```bash
GEMINI_API_URL=https://your-api-gateway/v1/chat/completions
GEMINI_API_KEY=your-api-key
```

## HuggingFace（可选，上传/下载数据集时需要）

```bash
HF_TOKEN=hf_xxx
```

## Sentence Transformer（可选，MQA 评估语义匹配）

默认从 HuggingFace Hub 下载 `Qwen/Qwen3-Embedding-0.6B`。如需指定本地路径：

```bash
SENTENCE_TRANSFORMER_MODEL=/path/to/your/model
```

## Demo 相关（仅运行 demo 时需要）

```bash
# 模型路径
MODEL_PATH=/path/to/modeling_qwen3_new_add_chunk
PROCESSOR_PATH=/path/to/modeling_minicpm26o_clean
MODELING_PATH=/path/to/modeling

# Checkpoint 路径
CHECKPOINT_PATH=/path/to/your_checkpoint.pt
DUPLEX_CHECKPOINT=/path/to/duplex_checkpoint.pt
TTS_CHECKPOINT=/path/to/tts_checkpoint.pt

# TTS 参考音频（声音克隆）
TTS_REF_AUDIO=/path/to/reference_audio.wav

# Token2wav
TOKEN2WAV_PATH=/path/to/Step-Audio2-main
TOKEN2WAV_MODEL_DIR=/path/to/token2wav
```

## .env 文件示例

在项目根目录创建 `.env` 文件：

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

程序会按以下顺序搜索 `.env` 文件：
1. 项目根目录 `.env`
2. 当前工作目录 `.env`
3. 当前工作目录 `.env.local`

已设置的环境变量不会被 `.env` 文件覆盖。
