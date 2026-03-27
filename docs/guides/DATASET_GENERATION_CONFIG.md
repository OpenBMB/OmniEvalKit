# Dataset Generation Configuration Guide

## Overview

The dataset generation configuration system allows you to customize prompt templates and generation parameters for different datasets, supporting both ASR and QA task types.

## Configuration File Structure

Configuration files are in JSON format with three main sections:

```json
{
    "asr_configs": {
        // ASR dataset configurations
    },
    "qa_configs": {
        // QA dataset configurations
    },
    "default_config": {
        // Default configuration (used when no dataset-specific config is found)
    }
}
```

### Configuration Fields

Each dataset configuration contains three fields:

- `user_prompt`: User prompt template, supports `{audio}` and `{question}` placeholders
- `system_prompt`: System prompt (optional, usually an empty string)
- `max_tokens`: Maximum number of generation tokens

## Default Configuration

The system provides a complete default configuration file: `o_e_Kit/configs/dataset_generation_configs.json`

### Default ASR Dataset Configurations

Includes 22 ASR dataset configurations:

**Chinese datasets:**
- wenetspeech_test_net/meeting
- aishell1/2/3_test
- commonvoice_zh/yue
- fleurs_zh

**English datasets:**
- gigaspeech_test
- librispeech_test/dev_clean/other
- commonvoice_en
- voxpopuli_en
- peoples_speech_test
- spgispeech_test
- tedlium1/2/3_test
- fleurs_en

### Default QA Dataset Configurations

Includes 10 QA dataset configurations:

- audioqa1m
- voicebench_alpacaeval
- voicebench_bbh
- voicebench_mmsu
- voicebench_openbookqa
- voicebench_advbench
- voicebench_commoneval
- voicebench_ifeval
- voicebench_sdqa
- voicebench_wildvoice

## Usage

### 1. Using the Default Configuration

No extra arguments needed — the system loads the default configuration automatically:

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_gigaspeech_test
```

### 2. Using a Custom Configuration

Specify a custom configuration file via the `--dataset_generation_config_path` argument:

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --dataset_generation_config_path path/to/custom_config.json \
    --eval_gigaspeech_test
```

### 3. Creating a Custom Configuration

Refer to the example file `o_e_Kit/configs/custom_generation_config_example.json`:

```json
{
    "asr_configs": {
        "gigaspeech_test": {
            "user_prompt": "Transcribe the following audio with precision:\n{audio}",
            "system_prompt": "You are an expert transcriptionist.",
            "max_tokens": 100
        }
    },
    "qa_configs": {
        "voicebench_alpacaeval": {
            "user_prompt": "Listen carefully and provide a comprehensive answer:\n{audio}",
            "system_prompt": "You are a helpful assistant.",
            "max_tokens": 300
        }
    },
    "default_config": {
        "user_prompt": "Process the audio:\n{audio}",
        "system_prompt": "",
        "max_tokens": 150
    }
}
```

## Multilingual Support

The system supports prompts in multiple languages by default:

- **Chinese**: "请细心听取音频内容，并将其准确转写出来。"
- **English**: "Please listen to the audio carefully and transcribe it with high precision."
- **Cantonese**: "請細心聽取音頻內容，並將其準確轉寫出來。"
- **French**: "Veuillez écouter attentivement l'audio et le transcrire avec précision."

## Notes

1. If the specified configuration file does not exist, the system falls back to a built-in minimal configuration.
2. If a particular dataset has no configuration, the `default_config` is used.
3. The `{audio}` placeholder is replaced with the audio token.
4. The `{question}` placeholder is used for QA tasks (when question text is available).

## Adding a New Dataset

Steps to add a new dataset configuration:

1. Add a new entry in the appropriate section (`asr_configs` or `qa_configs`) of the configuration file.
2. Use the dataset name as the key (must match the name in `dataset_args.py`).
3. Set appropriate prompt and max_tokens values.

For example:

```json
"new_dataset_name": {
    "user_prompt": "Your custom prompt with {audio}",
    "system_prompt": "",
    "max_tokens": 128
}
```

## Debugging

To see which configuration is actually used, check the log at model initialization:

```
Loaded dataset generation configs from: path/to/config.json
```

or

```
Using default dataset generation configs
```
