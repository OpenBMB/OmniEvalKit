# 环境配置与安装指南

本指南将为您提供详细的步骤说明，帮助您配置运行 Omni-Eval Kit 所需的环境并准备相关数据。

## 1. 环境配置

### 系统需求

- **Python**: 推荐使用 3.10 或更高版本。
- **PyTorch**: 请确保您安装了与您的 CUDA 工具包兼容的 PyTorch 版本。本项目已在 PyTorch 2.x 版本上测试通过。
- **Git**: 用于克隆本代码仓库。

### 安装步骤

1.  **克隆代码库**

    ```bash
    git clone https://github.com/OpenBMB/OmniEvalKit.git
    cd omnievalkit
    ```

2.  **安装 PyTorch**

    OmniEvalKit 依赖 PyTorch，请先根据你的 CUDA 版本安装：

    ```bash
    # 参考 https://pytorch.org/get-started/locally/ 选择合适的版本
    pip install torch
    ```

3.  **安装依赖库**

    **推荐方式：使用 [uv](https://docs.astral.sh/uv/)（更快）**

    ```bash
    # 安装 uv（如果尚未安装）
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 一键安装所有依赖（自动创建虚拟环境）
    uv sync --all-extras

    # 运行评估时使用 uv run，无需手动激活虚拟环境
    uv run torchrun --nproc_per_node=1 eval_main.py ...
    ```

    **使用 pip**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[all]"
    ```

    **按需安装可选依赖**

    项目依赖按功能分组，您可以只安装需要的部分：

    ```bash
    pip install -e .              # 仅核心依赖（ASR、Caption 等评测）
    pip install -e ".[api]"       # + API 调用（OpenAI、DashScope、Azure）
    pip install -e ".[embedding]" # + 句向量模型（sentence-transformers）
    pip install -e ".[dev]"       # + 开发工具（pytest、pre-commit）
    pip install -e ".[all]"       # 安装全部
    ```

## 2. 数据集准备

评测数据集托管在 HuggingFace 上，使用下载脚本可一键获取并自动还原为框架所需的目录结构。

### 快速下载

```bash
# 下载全部数据集（音频+图片+标注，约 108GB）
python scripts/hf_download.py --output_dir ./data

# 或只下载指定数据集
python scripts/hf_download.py --datasets gigaspeech_test,omnibench --output_dir ./data

# 查看可用数据集列表
python scripts/hf_download.py --list
```

详细说明请参阅 **[数据集下载指南](./DATA_DOWNLOAD_zh.md)**。

### 目录结构

下载完成后，`data/` 目录会自动生成如下结构：

```
data/
├── audio/
│   ├── asr/              # 语音识别数据集
│   │   ├── gigaspeech/
│   │   │   ├── test_files/    # 音频文件
│   │   │   └── test.jsonl     # 标注文件
│   │   ├── librispeech/
│   │   ├── wenetspeech/
│   │   ├── aishell1/
│   │   ├── commonvoice/
│   │   └── ...
│   ├── qa/               # 音频问答数据集
│   │   ├── voicebench/
│   │   └── ...
│   ├── caption/          # 音频描述数据集
│   ├── cls/              # 音频分类数据集
│   └── multitask/        # 多任务音频理解
└── omni/                 # 全模态数据集
```

### 支持的数据集

完整的数据集列表请参阅 HuggingFace 数据集仓库：**[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

涵盖 ASR（语音识别）、QA（问答）、多任务音频理解、音频描述与分类、全模态理解等多种任务类型。

### 标注文件格式

所有数据集统一使用 **JSONL 格式**（每行一个 JSON 对象）。详细格式规范请参阅 [音频数据集指南](./AUDIO_DATASETS_zh.md)。

### 覆盖默认路径

您可以通过命令行参数覆盖数据集路径，无需修改任何配置文件：

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_gigaspeech_test \
    --gigaspeech_test_data_prefix_dir /my/custom/path/audio/ \
    --gigaspeech_test_annotation_path /my/custom/path/test.jsonl
```

## 3. 配置 LLM 评估 API（推荐）

部分数据集（选择题、开放问答等）在评估时需要调用外部 LLM API 来提取或评判答案。在项目根目录创建 `.env` 文件：

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

支持任何 OpenAI 兼容的 API 端点（Azure OpenAI、vLLM、第三方转发等）。ASR 和 Caption 类数据集不需要此配置。

详细说明请参阅 **[LLM 评估配置指南](./LLM_EVALUATION_zh.md)**。

## 4. 下一步

环境配置完成后，请阅读 **[使用指南](./USAGE_zh.md)** 了解如何运行评估。
