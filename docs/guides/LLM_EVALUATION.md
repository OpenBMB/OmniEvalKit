# LLM Evaluation Configuration Guide

When automatically scoring model outputs, some datasets require calling an external LLM API (OpenAI-compatible endpoint) to extract or judge answers. This document explains how the evaluation pipeline works, which datasets need API access, and how to configure it properly.

## Evaluation Strategy Overview

The evaluation framework uses a **three-tier strategy**, trying each level in order:

```
Model prediction output
    │
    ▼
┌─────────────────────────┐
│ Tier 1: Rule Matching    │  Regex, template matching, exact match
│ (all evaluators)         │  Zero cost, fastest
└────────┬────────────────┘
         │ match failed
         ▼
┌─────────────────────────┐
│ Tier 2: Semantic Match   │  Sentence Transformer cosine similarity
│ (MQA only)               │  Requires GPU, optional
└────────┬────────────────┘
         │ match failed
         ▼
┌─────────────────────────┐
│ Tier 3: LLM Fallback    │  Calls OpenAI-compatible API
│ (requires API Key)       │  High accuracy, has API cost
└─────────────────────────┘
```

When Tier 1 (and Tier 2) can produce a **perfect score** determination, that result is used directly without calling the LLM. The LLM is only triggered as a fallback when rules cannot determine the answer.

## Quick Setup

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

`OPENAI_API_BASE` supports any API endpoint compatible with the OpenAI Chat Completions format, including:

- OpenAI official API
- Azure OpenAI
- Third-party API proxies (e.g., API2D, OpenRouter)
- Self-hosted inference services (e.g., vLLM, Ollama), as long as they provide a `/v1/chat/completions` endpoint

After configuration, run the built-in test script to verify connectivity:

```bash
python o_e_Kit/utils/metrics/llm_call_new.py
```

This script tests all configured API keys and reports their availability.

## Default Models

| Purpose | Default Model | Environment Variable Override |
|---------|--------------|-------------------------------|
| MQA answer extraction | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| RefQA answer matching | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| OpenQA scoring | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| Duplex/Caption evaluation | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| LiveSports LLM Judge | `gpt-4o` | `MODEL_GPT_4O` |

If your API endpoint uses different model names, override them via environment variables:

```bash
# Example: using DeepSeek as the evaluation model
MODEL_GPT_4O_MINI=deepseek-chat
MODEL_GPT_4O=deepseek-chat
```

## Datasets and Evaluation Methods

### Datasets That Do NOT Require LLM API

These datasets use pure rule-based evaluation and **work without API configuration**:

| Evaluation Method | Datasets | Metric |
|-------------------|----------|--------|
| WER (Word Error Rate) | gigaspeech_test, librispeech_test_clean/other, commonvoice_en, voxpopuli_en, fleurs_en, peoples_speech_test, tedlium3_test, commonvoice_fr | WER |
| CER (Character Error Rate) | wenetspeech_test_net/meeting, commonvoice_zh/yue, aishell1/2_test, kespeech_test, fleurs_zh | CER |
| Caption (BLEU/METEOR) | audiocaps_test, clothocaption_test, wavcaps_*, covost2_zh_en/en_zh | BLEU/METEOR/CIDEr |
| Safety evaluation | voicebench_advbench | Refusal rate |
| Instruction following | voicebench_ifeval | Follow rate |
| MCQ rule matching | OVOBench | Accuracy |
| StreamingBench | StreamingBench_REAL/OMNI/SQA | Accuracy |
| Event localization | ovavel | Frame/segment/event F1 |

### Datasets That Require LLM API

The following datasets call the LLM when rule matching fails. **Configuring API is recommended** for accurate evaluation:

| Evaluation Method | Datasets | Description |
|-------------------|----------|-------------|
| **MQA** (multiple-choice extraction) | voicebench_mmsu, voicebench_openbookqa, voice_cmmlu, mmau_test_mini, mmsu_bench, mmar_bench, daily_omni, omnibench, worldsense, av_odyssey, videomme, videomme_short, unobench_mc, ovobench, video_holmes, avut_benchmark_human/gemini, streamingbench_real/omni_fix/sqa, jointavbench, futureomni, avmeme_full/main | LLM extracts option letter from free text |
| **RefQA** (reference answer QA) | voicebench_sdqa, voicebench_bbh, audio_web_questions, audio_trivia_qa, vocalsound, meld, unobench | LLM judges if prediction is semantically consistent with reference answer |
| **OpenQA** (open-ended QA) | voicebench_alpacaeval, voicebench_alpacaeval_full, voicebench_commoneval, voicebench_wildvoice | LLM scores answer quality on a 1-5 scale |
| **LLM Judge** (A/B comparison) | livesports3k_cc | LLM compares model output vs. baseline (GPT-4o), computes win rate |
| **Omni LLM** | VisionCap, OmniCap, LiveCC, AVEvent, etc. (Duplex datasets) | LLM scores description/commentary quality |

## What Happens Without API Configuration?

If `OPENAI_API_KEY` is not set:

1. **ASR / Caption / Safety / OVAVEL (rule-based datasets)**: Not affected at all.
2. **MQA multiple-choice datasets**: Rule matching and Sentence Transformer still work and cover most samples; a few unresolvable samples are marked as `score=0, eval_method=failed`.
3. **OpenQA / LLM Judge datasets**: All LLM evaluations fail, scores will be 0.

We recommend configuring API at least for MQA / RefQA datasets to ensure evaluation accuracy.

## Multi-Key Rotation

When evaluating large volumes of data, a single API key may hit rate limits. The system supports configuring multiple backup keys that are automatically used when the primary key fails (insufficient balance, rate limit, etc.):

```bash
# .env
OPENAI_API_KEY=sk-main-key
OPENAI_API_KEY_1=sk-backup-key-1
OPENAI_API_KEY_2=sk-backup-key-2
OPENAI_API_KEY_3=sk-backup-key-3
```

Rotation logic:
- Primary key (`OPENAI_API_KEY`) is used first.
- When errors like insufficient balance, authentication failure, or rate limiting are detected, the system automatically switches to the next backup key.
- Each API call retries up to 5 times.

## Advanced Configuration

### Sentence Transformer Semantic Matching

The MQA evaluator has an additional Sentence Transformer semantic matching layer between rule matching and LLM. By default, `Qwen/Qwen3-Embedding-0.6B` is automatically downloaded from HuggingFace Hub. To specify a local model path:

```bash
SENTENCE_TRANSFORMER_MODEL=/path/to/your/embedding/model
```

### .env File Search Order

The program searches for `.env` files in the following order, stopping at the first one found:

1. Project root directory `.env`
2. Current working directory `.env`
3. Current working directory `.env.local`

Existing environment variables are **not** overridden by values in the `.env` file.

### Evaluation Report Statistics

After evaluation completes, the report shows evaluation method statistics:

- `rule_eval_count`: Number of samples successfully evaluated by rule matching
- `llm_eval_count`: Number of samples evaluated by LLM
- `failed_count`: Number of samples where both rules and LLM failed

If `failed_count` is high, check whether API configuration is correct.

## Complete .env Example

```bash
# === LLM Evaluation API (recommended) ===
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions

# Backup keys (optional)
OPENAI_API_KEY_1=sk-backup-1
OPENAI_API_KEY_2=sk-backup-2

# Custom model names (optional, when using non-OpenAI API)
# MODEL_GPT_4O_MINI=your-model-name
# MODEL_GPT_4O=your-model-name

# === Sentence Transformer (optional) ===
# SENTENCE_TRANSFORMER_MODEL=/path/to/local/model

# === HuggingFace (needed for dataset download) ===
# HF_TOKEN=hf_xxx
```

## Related Documentation

- **[Environment Variable Configuration](./CONFIGURATION.md)** — Full list of all environment variables
- **[Supported Evaluation Metrics](../reference/SUPPORTED_METRICS.md)** — Detailed description of each metric
- **[Usage Guide](./USAGE.md)** — How to run evaluations
