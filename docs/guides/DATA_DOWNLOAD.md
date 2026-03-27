# Data Download Guide

OmniEvalKit evaluation datasets are hosted on HuggingFace: **[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

This guide explains how to download and set up the data.

## Prerequisites

The download script dependencies (`huggingface_hub`, `pandas`, `pyarrow`, `tqdm`) are included in the project's core dependencies. If you have already followed the [Setup Guide](./SETUP.md), you're ready to go.

If you need to download gated datasets, log in to HuggingFace first:

```bash
huggingface-cli login
```

## Quick Start

### Download All Datasets

```bash
# Download all dataset audio, images, and metadata (~108GB)
python scripts/hf_download.py --output_dir ./data
```

After completion, the `data/` directory will automatically contain the required directory structure and JSONL annotation files.

### Download Specific Datasets

```bash
# Download only omnibench and daily_omni
python scripts/hf_download.py --datasets omnibench,daily_omni --output_dir ./data
```

### List Available Datasets

```bash
python scripts/hf_download.py --list
```

## Video Files

Since video files are large (~184GB), they are not included in the Parquet files. For datasets that contain videos, there are two ways to obtain them:

### Option 1: Automatic Download (Recommended)

```bash
python scripts/hf_download.py --output_dir ./data --download_videos
```

The script will attempt to automatically download video files from the original HuggingFace sources and place them in the correct locations.

### Option 2: Manual Download

When running the download script, it will display which datasets need videos and their sources:

```
Datasets requiring video download (5):
  - videomme → lmms-lab/Video-MME
  - daily_omni → DailyOmni/Daily-Omni
  ...
```

Download the video files from the corresponding HuggingFace repositories and place them in the appropriate `data/` subdirectories.

## Directory Structure

After downloading, the directory structure looks like this:

```
data/
├── audio/
│   ├── asr/
│   │   ├── gigaspeech/
│   │   │   ├── test.jsonl          # Annotation file
│   │   │   └── test_files/         # Audio files
│   │   ├── librispeech/
│   │   └── ...
│   ├── qa/
│   ├── caption/
│   └── ...
├── omni/
│   ├── raw_hf/
│   │   ├── omnibench/
│   │   │   ├── omnibench.jsonl     # Annotation file
│   │   │   └── mm_data/            # Audio + images
│   │   ├── daily-omni/
│   │   │   ├── daily_omni.jsonl
│   │   │   └── Videos/             # Requires separate download
│   │   └── ...
│   └── ...
```

## Verify Data

After downloading, you can run a quick evaluation to verify the data is ready:

```bash
# Test a single dataset
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_omnibench \
    --max_sample_num 5
```
