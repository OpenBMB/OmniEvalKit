# Environment Variable Configuration

This project uses environment variables to configure API keys, model paths, and other parameters. You can create a `.env` file in the project root directory, and the program will load it automatically.

## LLM Evaluation (Required for LLM Fallback Evaluation)

```bash
# OpenAI-compatible API
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions

# Multi-key rotation (configured by index)
OPENAI_API_KEY_1=sk-xxx
OPENAI_API_KEY_2=sk-yyy
```

When both rule matching and Sentence Transformer semantic matching fail to determine a result, the system automatically calls the LLM API for fallback evaluation.

## Gemini API (Optional, only needed when using the gemini_omni model)

```bash
GEMINI_API_URL=https://your-api-gateway/v1/chat/completions
GEMINI_API_KEY=your-api-key
```

## HuggingFace (Optional, needed for uploading/downloading datasets)

```bash
HF_TOKEN=hf_xxx
```

## Sentence Transformer (Optional, for MQA Semantic Matching)

By default, `Qwen/Qwen3-Embedding-0.6B` is downloaded from HuggingFace Hub. To specify a local path:

```bash
SENTENCE_TRANSFORMER_MODEL=/path/to/your/model
```

## Demo Related (Only needed when running demos)

```bash
# Model paths
MODEL_PATH=/path/to/modeling_qwen3_new_add_chunk
PROCESSOR_PATH=/path/to/modeling_minicpm26o_clean
MODELING_PATH=/path/to/modeling

# Checkpoint paths
CHECKPOINT_PATH=/path/to/your_checkpoint.pt
DUPLEX_CHECKPOINT=/path/to/duplex_checkpoint.pt
TTS_CHECKPOINT=/path/to/tts_checkpoint.pt

# TTS reference audio (voice cloning)
TTS_REF_AUDIO=/path/to/reference_audio.wav

# Token2wav
TOKEN2WAV_PATH=/path/to/Step-Audio2-main
TOKEN2WAV_MODEL_DIR=/path/to/token2wav
```

## .env File Example

Create a `.env` file in the project root directory:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

The program searches for `.env` files in the following order:
1. Project root directory `.env`
2. Current working directory `.env`
3. Current working directory `.env.local`

Existing environment variables will not be overridden by the `.env` file.
