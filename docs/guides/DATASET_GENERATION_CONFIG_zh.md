# 数据集生成配置指南

## 概述

数据集生成配置系统允许您为不同的数据集自定义prompt模板和生成参数，支持ASR和QA两类任务。

## 配置文件结构

配置文件为JSON格式，包含三个主要部分：

```json
{
    "asr_configs": {
        // ASR数据集配置
    },
    "qa_configs": {
        // QA数据集配置
    },
    "default_config": {
        // 默认配置（当找不到特定数据集配置时使用）
    }
}
```

### 配置项说明

每个数据集配置包含三个字段：

- `user_prompt`: 用户提示模板，支持 `{audio}` 和 `{question}` 占位符
- `system_prompt`: 系统提示（可选，通常为空字符串）
- `max_tokens`: 最大生成token数

## 默认配置

系统提供了完整的默认配置文件：`o_e_Kit/configs/dataset_generation_configs.json`

### ASR数据集默认配置

包含22个ASR数据集的配置：

**中文数据集：**
- wenetspeech_test_net/meeting
- aishell1/2/3_test
- commonvoice_zh/yue
- fleurs_zh

**英文数据集：**
- gigaspeech_test
- librispeech_test/dev_clean/other
- commonvoice_en
- voxpopuli_en
- peoples_speech_test
- spgispeech_test
- tedlium1/2/3_test
- fleurs_en

### QA数据集默认配置

包含10个QA数据集的配置：

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

## 使用方法

### 1. 使用默认配置

无需额外参数，系统会自动加载默认配置：

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_gigaspeech_test
```

### 2. 使用自定义配置

通过 `--dataset_generation_config_path` 参数指定自定义配置文件：

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --dataset_generation_config_path path/to/custom_config.json \
    --eval_gigaspeech_test
```

### 3. 创建自定义配置

参考示例文件 `o_e_Kit/configs/custom_generation_config_example.json`：

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

## 多语言支持

系统默认支持多种语言的prompt：

- **中文**: "请细心听取音频内容，并将其准确转写出来。"
- **英文**: "Please listen to the audio carefully and transcribe it with high precision."
- **粤语**: "請細心聽取音頻內容，並將其準確轉寫出來。"
- **法语**: "Veuillez écouter attentivement l'audio et le transcrire avec précision."

## 注意事项

1. 如果指定的配置文件不存在，系统会降级使用内置的最小配置
2. 如果某个数据集没有配置，会使用 `default_config`
3. `{audio}` 占位符会被替换为音频标记
4. `{question}` 占位符用于QA任务（如果有问题文本）

## 扩展新数据集

添加新数据集配置的步骤：

1. 在配置文件的相应部分（`asr_configs` 或 `qa_configs`）添加新配置
2. 使用数据集名称作为key（必须与 `dataset_args.py` 中的名称一致）
3. 设置合适的prompt和max_tokens

例如：

```json
"new_dataset_name": {
    "user_prompt": "Your custom prompt with {audio}",
    "system_prompt": "",
    "max_tokens": 128
}
```

## 调试

如果需要查看实际使用的配置，模型初始化时会打印：

```
Loaded dataset generation configs from: path/to/config.json
```

或

```
Using default dataset generation configs
```


