# Omni-Eval Kit (o_e_Kit)

[English](./README.md) | [中文说明](./README_zh.md)

**OmniEvalKit** is an evaluation framework designed for omni-modal large language models, with a focus on audio and audio-visual understanding. Based on OmniEvalKit, you can quickly reproduce benchmarks, implement your own models or datasets, and conduct fair comparisons with other open-source models. Our work [MiniCPM-o](https://huggingface.co/openbmb/MiniCPM-o-4_5) is evaluated using this framework.

## Key Features

-   **Distributed Evaluation**: Leverages `torch.distributed` and `torchrun` for efficient multi-GPU inference.
-   **Extensible Architecture**: Easily add new models, datasets, and evaluation metrics without modifying core code.
-   **Standardized Workflows**: Unified entry point (`eval_main.py`) and run scripts for all evaluation tasks.
-   **Rich Evaluation Metrics**: Built-in WER/CER, BLEU/METEOR/CIDEr, VQA scoring, LLM-as-judge, and more.
-   **Automated Reporting**: Built-in tools for generating evaluation reports.

## Quick Start

```bash
git clone https://github.com/OpenBMB/OmniEvalKit.git
cd omnievalkit

# 1. Install PyTorch first (choose the version matching your CUDA)
#    See https://pytorch.org/get-started/locally/
pip install torch

# 2. Install OmniEvalKit
# Recommended: using uv (much faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras

# Or using pip
pip install -e ".[all]"

# 3. Download evaluation datasets (see docs/guides/DATA_DOWNLOAD.md for details)
python scripts/hf_download.py --output_dir ./data
```

## Quick Example

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

Or use the provided example script:

```bash
bash scripts/example_eval.sh
```

To see all available arguments:

```bash
python eval_main.py --help
```

## Project Structure

```
omnievalkit/
├── eval_main.py                         # Main evaluation entry point
├── pyproject.toml                       # Project config & dependencies
├── requirement.txt
│
├── configs/                             # Configuration files
│   └── model_config/                    # Model configs (pool_step, chunk_length, etc.)
│
├── o_e_Kit/                             # Core framework package
│   ├── configs/                         # Internal configs
│   │   ├── duplex_configs.json          # Duplex mode generation config
│   │   └── generation_configs.json      # Generation parameter config
│   │
│   ├── datasets/                        # Dataset definitions & loading
│   │   ├── base_dataset.py              # Base dataset class
│   │   ├── audio_datasets.py            # Audio dataset registry
│   │   └── omni_datasets.py             # Omni-modal dataset registry
│   │
│   ├── models/                          # Model adapters
│   │   ├── minicpm/                     # MiniCPM-O (batch + duplex demo)
│   │   ├── qwen/                        # Qwen3-Omni
│   │   ├── gemini/                      # Gemini API
│   │   └── asr/                         # Whisper baseline
│   │
│   └── utils/                           # Utility modules
│       ├── get_args.py                  # Argument parsing entry
│       ├── model_loader.py              # Model loader
│       ├── dataloader.py                # Data loading & sharding
│       ├── dataset_loader.py            # Dataset discovery & loading
│       ├── infer.py                     # Inference engine (batch/chat/generate)
│       ├── eval.py                      # Evaluation dispatch
│       ├── evaluation_runner.py         # Evaluation orchestration
│       │
│       ├── args/                        # Argument definitions (modular)
│       │   ├── model_args.py            # Model arguments
│       │   ├── dataset_args.py          # Dataset registry & arguments
│       │   └── runtime_args.py          # Runtime arguments
│       │
│       ├── metrics/                     # Evaluation metrics
│       │   ├── evaluator_base.py        # Base evaluator (rule → ST → LLM fallback)
│       │   ├── wer_eval.py              # WER/CER (ASR)
│       │   ├── evaluator_mqa.py         # Multiple-choice QA
│       │   ├── evaluator_refqa.py       # Reference-answer QA
│       │   ├── evaluator_openqa.py      # Open-ended QA (LLM scoring)
│       │   ├── evaluator_caption.py     # Caption (BLEU/METEOR/CIDEr)
│       │   ├── llm_call_new.py          # OpenAI-compatible API client
│       │   └── ...                      # Safety, IFEval, StreamingBench, etc.
│       │
│       ├── text_normalization/          # Text normalization
│       └── logger/                      # Logging & progress
│
├── scripts/                             # Helper scripts
│   ├── example_eval.sh                  # Example evaluation launch script
│   ├── hf_download.py                   # Download datasets from HF
│   └── parquet_to_jsonl.py              # Parquet → JSONL conversion utility
│
├── docs/                                # Documentation
│   ├── guides/                          # User guides (setup, usage, data download, etc.)
│   ├── development/                     # Developer docs (architecture, contributing)
│   └── reference/                       # Reference (supported models, metrics, tasks)
```

## Datasets

All evaluation datasets are hosted on HuggingFace: **[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

## Documentation

### Guides

-   **[Setup and Installation Guide](./docs/guides/SETUP.md)** / [中文](./docs/guides/SETUP_zh.md)
-   **[Usage Guide](./docs/guides/USAGE.md)** / [中文](./docs/guides/USAGE_zh.md)
-   **[CLI Arguments](./docs/guides/ARGUMENTS.md)** / [中文](./docs/guides/ARGUMENTS_zh.md)
-   **[Data Download Guide](./docs/guides/DATA_DOWNLOAD.md)** / [中文](./docs/guides/DATA_DOWNLOAD_zh.md)
-   **[LLM Evaluation Configuration](./docs/guides/LLM_EVALUATION.md)** / [中文](./docs/guides/LLM_EVALUATION_zh.md)
-   **[Environment Variables](./docs/guides/CONFIGURATION.md)** / [中文](./docs/guides/CONFIGURATION_zh.md)

### Architecture & Development

-   **[Framework Architecture](./docs/development/ARCHITECTURE.md)** / [中文](./docs/development/ARCHITECTURE_zh.md)
-   **[Roadmap](./docs/ROADMAP.md)** / [中文](./docs/ROADMAP_zh.md)

### Reference

-   **[Supported Tasks](./docs/reference/SUPPORTED_TASKS.md)** / [中文](./docs/reference/SUPPORTED_TASKS_zh.md)
-   **[Supported Models](./docs/reference/SUPPORTED_MODELS.md)** / [中文](./docs/reference/SUPPORTED_MODELS_zh.md)
-   **[Supported Metrics](./docs/reference/SUPPORTED_METRICS.md)** / [中文](./docs/reference/SUPPORTED_METRICS_zh.md)

### Contributing

-   **[How to Contribute a New Model](./docs/development/CONTRIBUTING_MODELS.md)** / [中文](./docs/development/CONTRIBUTING_MODELS_zh.md)
-   **[How to Contribute a New Dataset](./docs/development/CONTRIBUTING_DATASETS.md)** / [中文](./docs/development/CONTRIBUTING_DATASETS_zh.md)
-   **[How to Contribute a New Evaluation Method](./docs/development/CONTRIBUTING_EVALS.md)** / [中文](./docs/development/CONTRIBUTING_EVALS_zh.md)
