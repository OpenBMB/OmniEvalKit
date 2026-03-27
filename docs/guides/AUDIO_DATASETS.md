# Audio Dataset Guide

## Overview

This framework supports loading and processing a wide range of mainstream audio datasets. Through the unified `AudioEvalDataset` class, it can automatically recognize and handle different annotation file formats.

## Supported Datasets

For the full list of supported datasets, see the HuggingFace dataset repository: **[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

Covers ASR (speech recognition), QA (question answering), multi-task audio understanding (MMAU, MMSU, MMAR), Caption (audio captioning), Classification (audio classification), Emotion (emotion recognition), and more.

## Annotation Format Requirements

The framework uses **JSONL format** (JSON Lines, one JSON object per line) to simplify data processing.

### Required Fields for ASR Tasks
- **WavPath** — Audio file path (relative to data_prefix_dir)
- **text** or **sentence** — Transcription text

### Required Fields for QA Tasks
- **WavPath** — Audio file path (relative to data_prefix_dir)
- **question** — Question text
- **answer** — Answer text
- **choices** — List of options (optional, for multiple-choice)

### Example Formats

#### ASR Task Format
```json
{"WavPath": "test/S0770/BAC009S0770W0259.wav", "text": "这是一段中文语音"}
{"WavPath": "common_voice_en_123.mp3", "sentence": "This is English speech"}
```

#### QA Task Format
```json
{"WavPath": "audio/test_001.wav", "question": "What sound is this?", "answer": "Dog barking", "choices": ["Dog barking", "Cat meowing", "Bird chirping", "Car horn"]}
```

#### Caption Task Format
```json
{"WavPath": "audio/test_001.wav", "caption": "A dog is barking loudly in the background"}
```

## Usage

### 1. Download Datasets

```bash
# One-click download and restore from HuggingFace
python scripts/hf_download.py --output_dir ./data

# See docs/guides/DATA_DOWNLOAD.md for details
```

### 2. Load a Dataset

```python
from o_e_Kit.datasets.audio_datasets import AudioEvalDataset

# Load GigaSpeech dataset
dataset = AudioEvalDataset(
    annotation_path='./data/gigaspeech/test.jsonl',
    data_prefix_dir='./data/gigaspeech/test_files/',
    dataset_name='gigaspeech_test'
)

# Get a sample
idx, paths, annotation = dataset[0]
print(f"Audio path: {paths['audio_path']}")
print(f"Transcription: {annotation['gt_answer']}")
```

### 3. Use in the Evaluation Framework

Datasets are registered in `o_e_Kit/utils/args/dataset_args.py` and can be enabled via command-line arguments:

```bash
# Evaluate ASR datasets
python eval_main.py --eval_gigaspeech_test --eval_wenetspeech_test_net

# Evaluate multi-task audio understanding datasets
python eval_main.py --eval_mmau_test_mini --eval_mmsu_bench --eval_mmar_bench

# Evaluate all audio datasets
python eval_main.py --eval_all_audio
```

## Adding a New Dataset

### Step 1: Prepare a JSONL Annotation File

Ensure your annotation file is in JSONL format, with each line containing:
- **WavPath** — Audio file path
- **text** or **sentence** — Transcription text

If the original data is not in JSONL format, convert it first.

### Step 2: Place Data Files

Place data files in the appropriate subdirectory under `data/`:

```bash
mkdir -p data/audio/asr/my_dataset
cp /path/to/your/audio_files data/audio/asr/my_dataset/test_files/
cp /path/to/your/annotation.jsonl data/audio/asr/my_dataset/test.jsonl
```

### Step 3: Register in dataset_args.py

```python
DatasetConfig(
    name="new_dataset_test",
    display_name="New Dataset Test",
    category="audio",
    subcategory="asr",  # or "qa", "caption", "cls", "emotion", etc.
    paths={
        "data_prefix_dir": "./data/new_dataset/test/",
        "annotation_path": "./data/new_dataset/test.jsonl"
    },
    description="Description of the new dataset"
),
```

### Step 4: Add Test Config in audio_datasets.py (Optional)

Add a test configuration to the `test_configs` list in `o_e_Kit/datasets/audio_datasets.py`:

```python
{
    'name': 'New Dataset Test',
    'annotation_path': './data/new_dataset/test.jsonl',
    'data_prefix_dir': './data/new_dataset/test/',
    'dataset_name': 'new_dataset_test'
},
```

## Dataset Format Examples

### LibriSpeech (.txt format)
```
103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED
103-1240-0001 MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD
```

### CommonVoice (.tsv format)
```
client_id	path	sentence	up_votes	down_votes	age	gender	accent	locale	segment
client123	common_voice_en_123.mp3	Hello world	2	0	twenties	male	us	en	train
```

### AISHELL (.txt format)
```
BAC009S0002W0122	而 对 楼市 成交 抑制 作用 最 大 的 限 购
BAC009S0002W0123	也 成为 地方 政府 的 眼中 钉
```

## FAQ

### Q: Dataset loading fails with "annotation file not found"
A: Check that the data has been downloaded. Run `ls -la ./data/` to confirm the files exist. To re-download, run `python scripts/hf_download.py --output_dir ./data`.

### Q: Audio file paths are incorrect
A: Verify the `data_prefix_dir` parameter is correct and that the concatenated path points to the audio file.

### Q: Annotation format is not recognized
A: Check the error message and verify that field names are in the supported list, or add a new field mapping.

## Completed Features

- [x] Support for mainstream ASR datasets (Chinese, English, multilingual)
- [x] Support for QA datasets (VoiceBench series, AudioQA1M, etc.)
- [x] Support for multi-task audio understanding datasets (MMAU, MMSU, MMAR)
- [x] Support for audio captioning datasets (AudioCaps, ClothoCaption, WavCaps)
- [x] Support for audio classification datasets (VocalSound)
- [x] Support for emotion recognition datasets (MELD)
- [x] Unified JSONL format processing
- [x] Automatic filtering of long audio (>30s)
- [x] Support for duration field to optimize loading speed

## Next Steps

- [ ] Add more dataset integration tests
- [ ] Support additional audio understanding datasets (ESC-50, AudioSet, etc.)
- [ ] Add dataset statistics and visualization features
- [ ] Support automatic dataset download and preprocessing
- [ ] Support streaming audio processing
- [ ] Add data augmentation features
