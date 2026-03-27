# Setup and Installation Guide

This guide provides step-by-step instructions for setting up the environment and preparing the necessary data to run evaluations with the Omni-Eval Kit.

## 1. Environment Setup

### Prerequisites

- **Python**: Version 3.10 or higher is recommended.
- **PyTorch**: Ensure you have a version of PyTorch compatible with your CUDA toolkit. This project has been tested with PyTorch 2.x.
- **Git**: For cloning the repository.

### Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/OpenBMB/OmniEvalKit.git
    cd omnievalkit
    ```

2.  **Install PyTorch**

    OmniEvalKit requires PyTorch. Install the version matching your CUDA toolkit:

    ```bash
    # See https://pytorch.org/get-started/locally/ for the right command
    pip install torch
    ```

3.  **Install Dependencies**

    **Recommended: Using [uv](https://docs.astral.sh/uv/) (much faster)**

    ```bash
    # Install uv (if not already installed)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install all dependencies (auto-creates virtual environment)
    uv sync --all-extras

    # Run evaluations with uv run (no need to activate venv manually)
    uv run torchrun --nproc_per_node=1 eval_main.py ...
    ```

    **Using pip**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[all]"
    ```

    **Install only what you need**

    Dependencies are grouped by functionality:

    ```bash
    pip install -e .              # Core only (ASR, Caption evaluation)
    pip install -e ".[api]"       # + API calls (OpenAI, DashScope, Azure)
    pip install -e ".[embedding]" # + Sentence embeddings (sentence-transformers)
    pip install -e ".[dev]"       # + Dev tools (pytest, pre-commit)
    pip install -e ".[all]"       # Everything
    ```

## 2. Dataset Preparation

Evaluation datasets are hosted on HuggingFace. Use the download script to fetch and automatically restore the directory structure required by the framework.

### Quick Download

```bash
# Download all datasets (audio + images + annotations, ~108GB)
python scripts/hf_download.py --output_dir ./data

# Download specific datasets
python scripts/hf_download.py --datasets gigaspeech_test,omnibench --output_dir ./data

# List available datasets
python scripts/hf_download.py --list
```

For more details, see **[Data Download Guide](./DATA_DOWNLOAD_zh.md)**.

### Directory Structure

After downloading, the `data/` directory will be organized as follows:

```
data/
├── audio/
│   ├── asr/              # Speech recognition datasets
│   │   ├── gigaspeech/
│   │   │   ├── test_files/    # Audio files
│   │   │   └── test.jsonl     # Annotation file
│   │   ├── librispeech/
│   │   ├── wenetspeech/
│   │   ├── aishell1/
│   │   ├── commonvoice/
│   │   └── ...
│   ├── qa/               # Audio question answering datasets
│   │   ├── voicebench/
│   │   └── ...
│   ├── caption/          # Audio captioning datasets
│   ├── cls/              # Audio classification datasets
│   └── multitask/        # Multi-task audio understanding
└── omni/                 # Omni-modal datasets
```

### Supported Datasets

For the full list of supported datasets, see the HuggingFace dataset repository: **[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

Covers ASR (speech recognition), QA (question answering), multi-task audio understanding, audio captioning & classification, omni-modal understanding, and more.

### Annotation Format

All datasets use **JSONL format** (one JSON object per line). See [Audio Datasets Guide](./AUDIO_DATASETS_zh.md) for detailed format specifications.

### Overriding Default Paths

You can override dataset paths via CLI without modifying any config:

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_gigaspeech_test \
    --gigaspeech_test_data_prefix_dir /my/custom/path/audio/ \
    --gigaspeech_test_annotation_path /my/custom/path/test.jsonl
```

## 3. Configure LLM Evaluation API (Recommended)

Some datasets (multiple-choice, open-ended QA, etc.) require an external LLM API to extract or judge answers during evaluation. Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

Any OpenAI-compatible API endpoint is supported (Azure OpenAI, vLLM, third-party proxies, etc.). ASR and Caption datasets do not require this configuration.

For details, see **[LLM Evaluation Guide](./LLM_EVALUATION.md)**.

## 4. Next Steps

After completing the setup, proceed to the **[Usage Guide](./USAGE.md)** to learn how to run evaluations.
