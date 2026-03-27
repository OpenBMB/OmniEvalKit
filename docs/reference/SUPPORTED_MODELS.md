# Supported Models

This document provides a detailed introduction to the model wrapper classes integrated in Omni-Eval Kit (o_e_Kit), including their capabilities, use cases, and invocation methods.

## 1. `MiniCPM_o` (MiniCPM-O Unified Model)

-   **File path**: `o_e_Kit/models/minicpm/minicpmo.py`
-   **Model type string**: `"minicpmo"`

### Capabilities & Use Cases

`MiniCPM_o` is the unified evaluation model wrapper for the MiniCPM-O series, supporting both batch and chat inference modes. It is suitable for ASR (speech recognition), audio QA, multi-modal understanding, and other evaluation tasks.

### Core Methods

#### `generate_batch(self, **batch)`

Batch generation method for running inference on multiple samples at once.

-   **Invocation**: Called when `--generate_method` is set to `"batch"`.
-   **Input (`batch` dict)**:
    -   `wav_paths: list[str]`: List of audio file paths.
    -   `questions: list[str]`: List of text questions corresponding to each audio.
    -   `datasetname: str`: Dataset name, used to look up the corresponding prompt.
-   **Output**: `list[str]` — Model predictions for each input.

#### `generate_chat(self, **batch)`

Chat-style generation method, processing samples one at a time.

-   **Invocation**: Called when `--generate_method` is set to `"chat"`.

#### `generate(self, **batch)`

General-purpose generation method.

-   **Invocation**: Called when `--generate_method` is set to `"generate"`.

## 2. `OmniDuplex` (Duplex Model)

-   **File path**: `o_e_Kit/models/minicpm/demo/duplex_runner.py`
-   **Model type string**: `"minicpmo_duplex_demo"`

### Capabilities & Use Cases

`OmniDuplex` is a model wrapper designed for **duplex or streaming** interaction tasks. It simulates real-time conversation scenarios where the model receives audio streams while simultaneously thinking and generating responses.

## 3. `Whisper` (ASR Baseline Model)

-   **File path**: `o_e_Kit/models/asr/whisper.py`
-   **Model type string**: `"whisper"`

### Capabilities & Use Cases

`Whisper` is the evaluation wrapper for OpenAI's Whisper model, used as a baseline model for ASR tasks.

## 4. `Qwen3OmniEvalModel` (Qwen3-Omni Multi-modal Model)

-   **File path**: `o_e_Kit/models/qwen/qwen3_omni.py`
-   **Model type string**: `"qwen3_omni"`

### Capabilities & Use Cases

`Qwen3OmniEvalModel` is the evaluation wrapper for the Qwen3-Omni multi-modal understanding model, using a unified `generate` inference interface.

## 5. `GeminiOmniApiEvalModel` (Gemini API Evaluation Model)

-   **File path**: `o_e_Kit/models/gemini/gemini_omni_api.py`
-   **Model type string**: `"gemini_omni"`

### Capabilities & Use Cases

`GeminiOmniApiEvalModel` evaluates via the Gemini API through an OpenAI-compatible gateway, using a unified `generate` inference interface.
