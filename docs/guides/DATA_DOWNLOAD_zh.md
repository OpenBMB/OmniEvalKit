# 数据集下载指南

OmniEvalKit 的评测数据集托管在 HuggingFace 上：**[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

本指南介绍如何一键下载并配置数据。

## 前提条件

下载脚本所需的依赖（`huggingface_hub`、`pandas`、`pyarrow`、`tqdm`）已包含在项目核心依赖中。如果您已按照 [环境配置指南](./SETUP_zh.md) 完成安装，可以直接使用。

如果需要下载受限数据集，请先登录 HuggingFace：

```bash
huggingface-cli login
```

## 快速开始

### 下载全部数据集

```bash
# 下载所有数据集的音频、图片和元数据（约 108GB）
python scripts/hf_download.py --output_dir ./data
```

执行完成后，`data/` 目录下会自动生成框架所需的目录结构和 JSONL 标注文件。

### 下载指定数据集

```bash
# 只下载 omnibench 和 daily_omni
python scripts/hf_download.py --datasets omnibench,daily_omni --output_dir ./data
```

### 查看可用数据集

```bash
python scripts/hf_download.py --list
```

## 视频文件

由于视频文件体积较大（约 184GB），Parquet 中不包含视频文件。对于包含视频的数据集，有两种获取方式：

### 方式一：自动下载（推荐）

```bash
python scripts/hf_download.py --output_dir ./data --download_videos
```

脚本会尝试从原始 HuggingFace 来源自动下载视频文件并放置到正确位置。

### 方式二：手动下载

运行下载脚本时，会提示需要视频的数据集及其来源：

```
需要额外下载视频的数据集 (5):
  - videomme → lmms-lab/Video-MME
  - daily_omni → DailyOmni/Daily-Omni
  ...
```

按照提示从对应的 HuggingFace 仓库下载视频文件，放到 `data/` 对应子目录下即可。

## 目录结构

下载完成后的目录结构：

```
data/
├── audio/
│   ├── asr/
│   │   ├── gigaspeech/
│   │   │   ├── test.jsonl          # 标注文件
│   │   │   └── test_files/         # 音频文件
│   │   ├── librispeech/
│   │   └── ...
│   ├── qa/
│   ├── caption/
│   └── ...
├── omni/
│   ├── raw_hf/
│   │   ├── omnibench/
│   │   │   ├── omnibench.jsonl     # 标注文件
│   │   │   └── mm_data/            # 音频+图片
│   │   ├── daily-omni/
│   │   │   ├── daily_omni.jsonl
│   │   │   └── Videos/             # 需单独下载
│   │   └── ...
│   └── ...
```

## 验证数据

下载完成后，可以直接运行评测验证数据是否就绪：

```bash
# 测试单个数据集
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_omnibench \
    --max_sample_num 5
```
