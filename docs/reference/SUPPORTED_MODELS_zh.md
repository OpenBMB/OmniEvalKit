# 已支持的模型功能介绍

本文档为开发者详细介绍了 Omni-Eval Kit (o_e_Kit) 中已集成的模型封装类，包括它们的功能、适用场景以及具体的调用方式。

## 1. `MiniCPM_o` (MiniCPM-O 统一模型)

-   **文件路径**: `o_e_Kit/models/minicpm/minicpmo.py`
-   **模型类型字符串**: `"minicpmo"`

### 功能与用途

`MiniCPM_o` 是 MiniCPM-O 系列的统一评测模型封装类，支持批处理和对话两种推理模式。它适用于 ASR（语音识别）、音频问答、多模态理解等多种评测任务。

### 核心方法

#### `generate_batch(self, **batch)`

批量生成方法，一次性对多条数据进行推理。

-   **调用方式**: 当 `--generate_method` 设置为 `"batch"` 时被调用。
-   **输入 (`batch` 字典)**:
    -   `wav_paths: list[str]`: 音频文件的路径列表。
    -   `questions: list[str]`: 与每段音频相对应的文本问题列表。
    -   `datasetname: str`: 数据集名称，用于查找对应的 Prompt。
-   **输出**: `list[str]` — 模型对每条输入的预测结果。

#### `generate_chat(self, **batch)`

对话式生成方法，逐条处理数据。

-   **调用方式**: 当 `--generate_method` 设置为 `"chat"` 时被调用。

#### `generate(self, **batch)`

通用生成方法。

-   **调用方式**: 当 `--generate_method` 设置为 `"generate"` 时被调用。

## 2. `OmniDuplex` (双工模型)

-   **文件路径**: `o_e_Kit/models/minicpm/demo/duplex_runner.py`
-   **模型类型字符串**: `"minicpmo_duplex_demo"`

### 功能与用途

`OmniDuplex` 是为**双工（Duplex）或流式（Streaming）** 交互任务设计的模型封装类。它模拟的是一个可以边接收音频流、边进行思考和生成的实时对话场景。

## 3. `Whisper` (ASR 基线模型)

-   **文件路径**: `o_e_Kit/models/asr/whisper.py`
-   **模型类型字符串**: `"whisper"`

### 功能与用途

`Whisper` 是 OpenAI Whisper 模型的评测封装类，作为 ASR 任务的基线模型使用。

## 4. `Qwen3OmniEvalModel` (Qwen3-Omni 多模态理解模型)

-   **文件路径**: `o_e_Kit/models/qwen/qwen3_omni.py`
-   **模型类型字符串**: `"qwen3_omni"`

### 功能与用途

`Qwen3OmniEvalModel` 是 Qwen3-Omni 多模态理解模型的评测封装类，统一使用 `generate` 推理接口。

## 5. `GeminiOmniApiEvalModel` (Gemini API 评测模型)

-   **文件路径**: `o_e_Kit/models/gemini/gemini_omni_api.py`
-   **模型类型字符串**: `"gemini_omni"`

### 功能与用途

`GeminiOmniApiEvalModel` 通过 OpenAI 兼容网关调用 Gemini API 进行评测，统一使用 `generate` 推理接口。
