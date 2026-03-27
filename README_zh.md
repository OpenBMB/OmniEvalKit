# Omni-Eval Kit (o_e_Kit)

[English](./README.md) | [中文说明](./README_zh.md)

**OmniEvalKit** 是一个专为全模态大语言模型设计的评测框架，聚焦于音频与视听理解。基于 OmniEvalKit，你可以快速复现 benchmark、接入自己的模型或数据集，并与其他开源模型进行公平对比。我们的工作 [MiniCPM-o](https://huggingface.co/openbmb/MiniCPM-o-4_5) 即使用本框架进行评测。

## 功能特性

-   **分布式评测**: 基于 `torch.distributed` 和 `torchrun`，支持高效的多 GPU 推理。
-   **可扩展架构**: 轻松添加新的模型、数据集或评测指标，无需修改核心代码。
-   **标准化流程**: 统一的入口（`eval_main.py`）和运行脚本，覆盖所有评测任务。
-   **丰富的评测指标**: 内置 WER/CER、BLEU/METEOR/CIDEr、VQA 评分、LLM-as-judge 等多种指标。
-   **自动化报告**: 内置评测报告生成工具。

## 快速开始

```bash
git clone https://github.com/OpenBMB/OmniEvalKit.git
cd omnievalkit

# 1. 先安装 PyTorch（选择与你的 CUDA 版本匹配的版本）
#    参考 https://pytorch.org/get-started/locally/
pip install torch

# 2. 安装 OmniEvalKit
# 推荐：使用 uv（安装速度更快）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# 或使用 pip
pip install -e ".[all]"

# 3. 下载评测数据集（详见 docs/guides/DATA_DOWNLOAD_zh.md）
python scripts/hf_download.py --output_dir ./data
```

## 快速示例

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/your/model \
    --answer_path ./results \
    --model_name my_minicpm_eval \
    --batchsize 4 \
    --max_sample_num 100 \
    --eval_gigaspeech_test
```

或使用提供的示例脚本：

```bash
bash scripts/example_eval.sh
```

查看所有可用参数：

```bash
python eval_main.py --help
```

## 项目结构

```
omnievalkit/
├── eval_main.py                         # 评测主入口
├── pyproject.toml                       # 项目配置与依赖
├── requirement.txt
│
├── configs/                             # 配置文件
│   └── model_config/                    # 模型配置（pool_step、chunk_length 等）
│
├── o_e_Kit/                             # 核心框架包
│   ├── configs/                         # 内部配置
│   │   ├── duplex_configs.json          # 双工模式生成配置
│   │   └── generation_configs.json      # 生成参数配置
│   │
│   ├── datasets/                        # 数据集定义与加载
│   │   ├── base_dataset.py              # 数据集基类
│   │   ├── audio_datasets.py            # 音频数据集注册
│   │   └── omni_datasets.py             # 全模态数据集注册
│   │
│   ├── models/                          # 模型适配器
│   │   ├── minicpm/                     # MiniCPM-O（批处理 + 双工 demo）
│   │   ├── qwen/                        # Qwen3-Omni
│   │   ├── gemini/                      # Gemini API
│   │   └── asr/                         # Whisper 基线
│   │
│   └── utils/                           # 工具模块
│       ├── get_args.py                  # 参数解析入口
│       ├── model_loader.py              # 模型加载器
│       ├── dataloader.py                # 数据加载与分片
│       ├── dataset_loader.py            # 数据集发现与加载
│       ├── infer.py                     # 推理引擎（batch/chat/generate）
│       ├── eval.py                      # 评测调度
│       ├── evaluation_runner.py         # 评测编排
│       │
│       ├── args/                        # 参数定义（模块化）
│       │   ├── model_args.py            # 模型参数
│       │   ├── dataset_args.py          # 数据集注册与参数
│       │   └── runtime_args.py          # 运行时参数
│       │
│       ├── metrics/                     # 评测指标
│       │   ├── evaluator_base.py        # 基础评估器（rule → ST → LLM 回退）
│       │   ├── wer_eval.py              # WER/CER（ASR）
│       │   ├── evaluator_mqa.py         # 多选题 QA
│       │   ├── evaluator_refqa.py       # 参考答案 QA
│       │   ├── evaluator_openqa.py      # 开放式 QA（LLM 评分）
│       │   ├── evaluator_caption.py     # Caption（BLEU/METEOR/CIDEr）
│       │   ├── llm_call_new.py          # OpenAI 兼容 API 客户端
│       │   └── ...                      # Safety、IFEval、StreamingBench 等
│       │
│       ├── text_normalization/          # 文本归一化
│       └── logger/                      # 日志与进度
│
├── scripts/                             # 辅助脚本
│   ├── example_eval.sh                  # 评测启动示例脚本
│   ├── hf_download.py                   # 从 HF 下载数据集
│   └── parquet_to_jsonl.py              # Parquet → JSONL 转换工具
│
├── docs/                                # 文档
│   ├── guides/                          # 用户指南（设置、使用、数据下载等）
│   ├── development/                     # 开发者文档（架构、贡献指南）
│   └── reference/                       # 参考手册（支持的模型、指标、任务）
```

## 数据集

所有评测数据集托管在 HuggingFace：**[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

## 文档

### 使用指南

-   **[设置与安装指南](./docs/guides/SETUP_zh.md)** / [English](./docs/guides/SETUP.md)
-   **[评测运行指南](./docs/guides/USAGE_zh.md)** / [English](./docs/guides/USAGE.md)
-   **[命令行参数详解](./docs/guides/ARGUMENTS_zh.md)** / [English](./docs/guides/ARGUMENTS.md)
-   **[数据集下载指南](./docs/guides/DATA_DOWNLOAD_zh.md)** / [English](./docs/guides/DATA_DOWNLOAD.md)
-   **[LLM 评估配置指南](./docs/guides/LLM_EVALUATION_zh.md)** / [English](./docs/guides/LLM_EVALUATION.md)
-   **[环境变量配置](./docs/guides/CONFIGURATION_zh.md)** / [English](./docs/guides/CONFIGURATION.md)

### 架构与开发

-   **[框架架构设计](./docs/development/ARCHITECTURE_zh.md)** / [English](./docs/development/ARCHITECTURE.md)
-   **[项目路线图与展望](./docs/ROADMAP_zh.md)** / [English](./docs/ROADMAP.md)

### 参考手册

-   **[当前支持的任务列表](./docs/reference/SUPPORTED_TASKS_zh.md)** / [English](./docs/reference/SUPPORTED_TASKS.md)
-   **[已支持的模型功能介绍](./docs/reference/SUPPORTED_MODELS_zh.md)** / [English](./docs/reference/SUPPORTED_MODELS.md)
-   **[已支持的评测指标](./docs/reference/SUPPORTED_METRICS_zh.md)** / [English](./docs/reference/SUPPORTED_METRICS.md)

### 开发者贡献

-   **[如何贡献一个新模型](./docs/development/CONTRIBUTING_MODELS_zh.md)** / [English](./docs/development/CONTRIBUTING_MODELS.md)
-   **[如何贡献一个新数据集](./docs/development/CONTRIBUTING_DATASETS_zh.md)** / [English](./docs/development/CONTRIBUTING_DATASETS.md)
-   **[如何贡献一个新的评测方法](./docs/development/CONTRIBUTING_EVALS_zh.md)** / [English](./docs/development/CONTRIBUTING_EVALS.md)
