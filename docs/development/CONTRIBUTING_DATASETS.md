# How to Contribute a New Dataset

This guide explains how to add a new evaluation dataset to the Omni-Eval Kit framework.

## Design Principles

- **Unified format**: All datasets should be converted to the framework's standard format (JSONL)
- **Config-driven**: Dataset configurations are managed centrally via `DATASET_REGISTRY`
- **Code reuse**: Prefer reusing existing `BaseDataset` subclasses over writing new ones
- **Evaluation integration**: Each dataset should have a corresponding evaluation method

## Overview

1. **Data format conversion** → 2. **Place data files** → 3. **Register config** → 4. **Select/implement dataset class** → 5. **Configure evaluation** → 6. **Test & verify**

## Step-by-Step Guide

### 1. Data Format Conversion

Convert your dataset to **JSONL** format (one JSON object per line). The framework automatically adapts to different field names.

#### Audio Dataset Format

**Required fields:**
- `WavPath`: Relative path to the audio file (relative to `data_prefix_dir`)

**Text fields (at least one required):**
- `text` / `sentence` / `transcription`: ASR transcription
- `caption`: Audio captioning
- `answer` / `answers`: QA task answers
- `reference`: Reference text
- `label` / `normalized_text`: Other text labels

**Optional fields:**
- `question` / `prompt`: Question or prompt (for QA tasks)
- `id` / `audio_id`: Sample ID

#### Format Examples

```json
{"WavPath": "audio/sample_001.wav", "text": "This is the transcription"}
{"WavPath": "audio/qa_001.wav", "question": "What is said in the audio?", "answer": "The correct answer"}
{"WavPath": "audio/cap_001.wav", "caption": "Birds chirping and water flowing"}
```

### 2. Place Data Files

Place your data files into the `data/` directory:

```bash
mkdir -p data/audio/asr/my_dataset
cp -r /path/to/original/audio/files data/audio/asr/my_dataset/audio_files
cp /path/to/converted/annotation.jsonl data/audio/asr/my_dataset/test.jsonl
```

### 3. Register in the Config Registry

Edit `o_e_Kit/utils/args/dataset_args.py` and add a new entry to `DATASET_REGISTRY`:

```python
DatasetConfig(
    name="my_dataset_asr",
    display_name="My Dataset ASR",
    category="audio",
    subcategory="asr",
    paths={
        "data_prefix_dir": "./data/audio/asr/my_dataset/audio_files/",
        "annotation_path": "./data/audio/asr/my_dataset/test.jsonl"
    },
    default_enabled=False,
    description="My dataset for ASR evaluation"
)
```

This automatically generates CLI arguments:
- `--eval_my_dataset_asr`: Enable evaluation
- `--my_dataset_asr_data_prefix_dir`: Data directory
- `--my_dataset_asr_annotation_path`: Annotation file

### 4. Use the Unified Dataset Class

**Audio datasets use `AudioEvalDataset` — no need to create a new class!**

`AudioEvalDataset` automatically:
- Recognizes multiple field name variants (text, sentence, transcription, caption, etc.)
- Infers task type (ASR, QA, Caption)
- Handles dataset-specific path quirks

Register it in `o_e_Kit/utils/dataset_loader.py`:

```python
def load_dataset(args, dataset_name):
    if dataset_name in ["my_dataset_asr"]:
        from o_e_Kit.datasets.audio_datasets import AudioEvalDataset
        return AudioEvalDataset(
            annotation_path=getattr(args, f"{dataset_name}_annotation_path"),
            data_prefix_dir=getattr(args, f"{dataset_name}_data_prefix_dir"),
            dataset_name=dataset_name
        )
```

For other modalities:
- **Video datasets**: Use `VideoEvalDataset`
- **Omni-modal datasets**: Use `OmniEvalDataset`
- **Duplex datasets**: Use `DuplexDataset`

### 5. Configure Evaluation Method

Register the dataset-to-evaluator mapping in `o_e_Kit/utils/evaluation_runner_audio.py`:

```python
def evaluate_dataset(dataset_name, predictions, args):
    if dataset_name in ["gigaspeech_test", "my_dataset_asr"]:
        from o_e_Kit.utils.metrics.wer_eval import WER_Eval
        evaluator = WER_Eval(lang='en', metric='wer')

    elif dataset_name in ["audioqa1m", "my_dataset_qa"]:
        from o_e_Kit.utils.metrics.evaluator_openqa import OpenQAEvaluator
        evaluator = OpenQAEvaluator()
```

Available evaluators:
- `WER_Eval`: Speech recognition (WER/CER)
- `OpenQAEvaluator`: Open-ended QA
- `CaptionEvaluator`: Caption generation (BLEU, etc.)
- `MCQEvaluator`: Multiple-choice questions
- `SafetyEvaluator`: Safety evaluation

### 6. Test & Verify

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/model \
    --eval_my_dataset_asr \
    --max_sample_num 10 \
    --batchsize 2
```

Verify that:
1. Data loads correctly
2. Inference runs without errors
3. Evaluation metrics are reasonable
4. Result files are generated

## FAQ

### Q: How to handle multiple annotation formats?
A: Unify during the data conversion step, or add compatibility logic in `__getitem__`.

### Q: Dataset is too large?
A: Use `--datasets` flag to download only specific datasets, support streaming loading, or create a subset for testing.

### Q: Need a custom evaluation metric?
A: See [How to Contribute a New Evaluation Method](CONTRIBUTING_EVALS.md) to create a new evaluator.
