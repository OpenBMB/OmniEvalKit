# LLM 评估配置指南

本项目在对模型输出进行自动评分时，部分数据集需要调用外部 LLM API（OpenAI 兼容接口）来提取或判定答案。本文档说明评估流程的工作原理、哪些数据集需要 API、以及如何正确配置。

## 评估策略概述

评估框架采用**三级策略**，逐级尝试：

```
模型预测输出
    │
    ▼
┌─────────────────────┐
│ 第1级：规则匹配       │  正则表达式、模板匹配、精确匹配
│ （所有评估器都具备）   │  零成本、速度最快
└────────┬────────────┘
         │ 匹配失败
         ▼
┌─────────────────────┐
│ 第2级：语义匹配       │  Sentence Transformer 余弦相似度
│ （仅 MQA 选择题）     │  需要 GPU，可选
└────────┬────────────┘
         │ 匹配失败
         ▼
┌─────────────────────┐
│ 第3级：LLM Fallback  │  调用 OpenAI 兼容 API
│ （需要配置 API Key）  │  准确率高，有 API 调用成本
└─────────────────────┘
```

当第1级（和第2级）能够给出**满分**判定时，直接采用该结果，不会调用 LLM。只有在规则无法确定答案时，才会触发 LLM 进行兜底评估。

## 快速配置

在项目根目录创建 `.env` 文件：

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

`OPENAI_API_BASE` 支持任何兼容 OpenAI Chat Completions 格式的 API 端点，包括：

- OpenAI 官方 API
- Azure OpenAI
- 第三方 API 转发服务（如 API2D、OpenRouter 等）
- 自建推理服务（如 vLLM、Ollama 等），只要提供 `/v1/chat/completions` 接口

配置完成后，可以运行内置测试脚本验证连通性：

```bash
python o_e_Kit/utils/metrics/llm_call_new.py
```

该脚本会逐一测试所有已配置的 API Key，并报告可用状态。

## 默认使用的模型

| 用途 | 默认模型 | 环境变量覆盖 |
|------|---------|-------------|
| 选择题答案提取 (MQA) | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| 参考答案问答 (RefQA) | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| 开放问答评分 (OpenQA) | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| Duplex/Caption 评估 | `gpt-4o-mini` | `MODEL_GPT_4O_MINI` |
| LiveSports LLM Judge | `gpt-4o` | `MODEL_GPT_4O` |

如果你使用的 API 端点提供的模型名称不同，可以通过环境变量覆盖：

```bash
# 例如使用 DeepSeek 作为评估模型
MODEL_GPT_4O_MINI=deepseek-chat
MODEL_GPT_4O=deepseek-chat
```

## 数据集与评估方式对照表

### 不需要 LLM API 的数据集

这些数据集使用纯规则评估，**无需配置 API** 即可正常工作：

| 评估方式 | 数据集 | 指标 |
|---------|--------|------|
| WER（词错误率） | gigaspeech_test, librispeech_test_clean/other, commonvoice_en, voxpopuli_en, fleurs_en, peoples_speech_test, tedlium3_test, commonvoice_fr | WER |
| CER（字错误率） | wenetspeech_test_net/meeting, commonvoice_zh/yue, aishell1/2_test, kespeech_test, fleurs_zh | CER |
| Caption（BLEU/METEOR） | audiocaps_test, clothocaption_test, wavcaps_*, covost2_zh_en/en_zh | BLEU/METEOR/CIDEr |
| 安全评估 | voicebench_advbench | 拒绝率 |
| 指令遵循 | voicebench_ifeval | 遵循率 |
| MCQ 规则匹配 | OVOBench | 准确率 |
| StreamingBench | StreamingBench_REAL/OMNI/SQA | 准确率 |
| 事件定位 | ovavel | 帧级/片段级/事件级 F1 |

### 需要 LLM API 的数据集

以下数据集在规则匹配失败时会调用 LLM，**建议配置 API** 以获得准确的评估结果：

| 评估方式 | 数据集 | 说明 |
|---------|--------|------|
| **MQA**（选择题提取） | voicebench_mmsu, voicebench_openbookqa, voice_cmmlu, mmau_test_mini, mmsu_bench, mmar_bench, daily_omni, omnibench, worldsense, av_odyssey, videomme, videomme_short, unobench_mc, ovobench, video_holmes, avut_benchmark_human/gemini, streamingbench_real/omni_fix/sqa, jointavbench, futureomni, avmeme_full/main | LLM 从自由文本中提取选项字母 |
| **RefQA**（参考答案问答） | voicebench_sdqa, voicebench_bbh, audio_web_questions, audio_trivia_qa, vocalsound, meld, unobench | LLM 判定预测是否与参考答案语义一致 |
| **OpenQA**（开放问答） | voicebench_alpacaeval, voicebench_alpacaeval_full, voicebench_commoneval, voicebench_wildvoice | LLM 对答案质量进行 1-5 分评分 |
| **LLM Judge**（A/B 评判） | livesports3k_cc | LLM 对比模型输出与基线（GPT-4o），计算胜率 |
| **Omni LLM** | VisionCap, OmniCap, LiveCC, AVEvent 等 Duplex 数据集 | LLM 对描述/解说质量评分 |

## 不配置 API 会怎样？

如果未设置 `OPENAI_API_KEY`：

1. **ASR / Caption / Safety / OVAVEL 等纯规则数据集**：完全不受影响
2. **MQA 选择题数据集**：规则匹配和 Sentence Transformer 仍可工作，能覆盖大部分样本；少量无法提取的样本会被标记为 `score=0, eval_method=failed`
3. **OpenQA / LLM Judge 数据集**：所有样本的 LLM 评估都会失败，分数全部为 0

建议至少为 MQA / RefQA 类数据集配置 API，以确保评估准确性。

## 多 Key 轮换

当评估数据量大时，单个 API Key 可能遇到速率限制。系统支持配置多个备选 Key，在主 Key 失败（余额不足、速率限制等）时自动切换：

```bash
# .env
OPENAI_API_KEY=sk-main-key
OPENAI_API_KEY_1=sk-backup-key-1
OPENAI_API_KEY_2=sk-backup-key-2
OPENAI_API_KEY_3=sk-backup-key-3
```

切换逻辑：
- 主 Key（`OPENAI_API_KEY`）优先使用
- 当检测到余额不足、认证失败、速率限制等错误时，自动切换到下一个备选 Key
- 每次 API 调用最多重试 5 次

## 高级配置

### Sentence Transformer 语义匹配

MQA 评估器在规则匹配和 LLM 之间还有一层 Sentence Transformer 语义匹配。默认使用 `Qwen/Qwen3-Embedding-0.6B`，会从 HuggingFace Hub 自动下载。如果需要指定本地模型路径：

```bash
SENTENCE_TRANSFORMER_MODEL=/path/to/your/embedding/model
```

### .env 文件搜索顺序

程序会按以下顺序搜索 `.env` 文件，找到第一个即停止：

1. 项目根目录 `.env`
2. 当前工作目录 `.env`
3. 当前工作目录 `.env.local`

已设置的环境变量**不会**被 `.env` 文件中的同名变量覆盖。

### 评估报告中的统计信息

评估完成后，报告会显示评估方法统计：

- `rule_eval_count`: 通过规则匹配成功评估的样本数
- `llm_eval_count`: 通过 LLM 评估的样本数
- `failed_count`: 评估失败的样本数（规则和 LLM 均失败）

如果 `failed_count` 较高，请检查 API 配置是否正确。

## 完整 .env 示例

```bash
# === LLM 评估 API（推荐配置）===
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions

# 备选 Key（可选）
OPENAI_API_KEY_1=sk-backup-1
OPENAI_API_KEY_2=sk-backup-2

# 自定义模型名称（可选，当使用非 OpenAI API 时）
# MODEL_GPT_4O_MINI=your-model-name
# MODEL_GPT_4O=your-model-name

# === Sentence Transformer（可选）===
# SENTENCE_TRANSFORMER_MODEL=/path/to/local/model

# === HuggingFace（下载数据集时需要）===
# HF_TOKEN=hf_xxx
```

## 相关文档

- **[环境变量配置](./CONFIGURATION_zh.md)** - 所有环境变量的完整列表
- **[已支持的评测指标](../reference/SUPPORTED_METRICS_zh.md)** - 各评估指标的详细说明
- **[使用指南](./USAGE_zh.md)** - 如何运行评估
