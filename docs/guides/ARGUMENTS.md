# Command-Line Arguments Reference

This document provides a detailed explanation of all command-line arguments available when running `eval_main.py`. Arguments are organized into three main categories using a modular design.

## Table of Contents

1. [Model Arguments](#1-model-arguments)
2. [Dataset Arguments](#2-dataset-arguments)
3. [Runtime Arguments](#3-runtime-arguments)
4. [Evaluation Control Arguments](#4-evaluation-control-arguments)
5. [Usage Examples](#5-usage-examples)

---

## 1. Model Arguments

Model-related arguments are defined in `o_e_Kit/utils/args/model_args.py`.

### Basic Configuration
-   `--model_type <type>`
    -   **Description**: Select the model type to use
    -   **Choices**: `minicpmo`, `minicpmo_duplex_demo`, `whisper`, `qwen3_omni`, `gemini_omni`
    -   **Default**: `minicpmo`

-   `--model_name <name>`
    -   **Description**: Model identifier used for naming result files
    -   **Default**: `minicpm26o`

-   `--generate_method <method>`
    -   **Description**: Select the inference method for the model
    -   **Choices**: `batch`, `chat`, `generate`
    -   **Default**: Automatically inferred based on model type

### Model Paths
-   `--model_path <path>`
    -   **Description**: Directory containing model files
    -   **Default**: Automatically set based on model type

-   `--tokenizer_path <path>`
    -   **Description**: Tokenizer path
    -   **Default**: Same as model path

-   `--pt_path <path>`
    -   **Description**: Pre-trained weights file path (.pt file)
    -   **Default**: `None`

-   `--config_path <path>`
    -   **Description**: Custom configuration file path
    -   **Default**: `None`

-   `--dataset_generation_config_path <path>`
    -   **Description**: Dataset generation config JSON file path for customizing per-dataset prompts and generation parameters
    -   **Default**: `None`

## 2. Dataset Arguments

Dataset arguments are managed through a configuration-driven approach, defined in `o_e_Kit/utils/args/dataset_args.py`.

### Auto-Generated Arguments

Each dataset registered in `DATASET_REGISTRY` automatically generates the following arguments:

- `--eval_<dataset_name>`: Enable evaluation for this dataset (boolean flag)
- `--<dataset_name>_data_prefix_dir`: Data file directory path
- `--<dataset_name>_annotation_path`: Annotation file path

### Example Dataset Arguments

**GigaSpeech Test Set:**
- `--eval_gigaspeech_test`: Enable GigaSpeech test set evaluation
- `--gigaspeech_test_data_prefix_dir`: Audio file directory
- `--gigaspeech_test_annotation_path`: Annotation file path

**AudioQA1M Dataset:**
- `--eval_audioqa1m`: Enable AudioQA1M evaluation
- `--audioqa1m_data_prefix_dir`: Data directory
- `--audioqa1m_annotation_path`: Annotation file

### Batch Evaluation Control
- `--eval_all`: Enable evaluation for all datasets
- `--eval_all_audio`: Enable all audio datasets
- `--eval_all_video`: Enable all video datasets
- `--eval_all_omni`: Enable all omni-modal datasets

## 3. Runtime Arguments

Runtime arguments are defined in `o_e_Kit/utils/args/runtime_args.py`.

### Batching and Sampling
- `--batchsize <size>`: Batch size (default: 2)
- `--max_sample_num <num>`: Maximum number of samples for quick testing (default: None)
- `--max_seq_len <len>`: Maximum sequence length (default: 8192)

### Output Settings
- `--answer_path <path>`: Results save path (default: `./results/`)
- `--save_interval <num>`: Save interval (default: 1000)

### Distributed Settings
- `--local_rank`: Local process rank for distributed training
- `--world_size`: Total number of processes
- `--master_port`: Master node port

### Device Configuration
- `--device`: Execution device (cuda/cpu)
- `--dtype`: Data type (fp16/bf16/fp32)

## 4. Evaluation Control Arguments

### Evaluation Modes
- `--eval_mode`: Evaluation mode (single/batch/streaming)
- `--eval_metrics`: List of evaluation metrics to use

### Special Configuration
- `--streaming_context_time`: StreamingBench context time (seconds)
- `--streaming_tasks`: StreamingBench task list
- `--livecc_data_type`: LiveCC data type (clipped/frames)

## 5. Usage Examples

### Basic Evaluation
```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/model \
    --eval_gigaspeech_test \
    --batchsize 4
```

### Override Dataset Paths
```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_audioqa1m \
    --audioqa1m_data_prefix_dir /custom/path/to/data/ \
    --audioqa1m_annotation_path /custom/path/to/ann.jsonl
```

### Batch Evaluate All Audio Datasets
```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_all_audio \
    --max_sample_num 100 \
    --answer_path ./test_results/
```

## View All Available Arguments

To see the full list of arguments and their current values:

```bash
python eval_main.py --help
```

## Configuration File Support

In addition to command-line arguments, you can manage parameters through configuration files:

1. **Model Config**: Specify a JSON model config via `--config_path`
2. **Dataset Paths**: Override default paths using environment variables
3. **Batch Config**: Create shell scripts with preset argument combinations

## Argument Priority

Argument values are resolved in the following order (highest to lowest priority):

1. Explicitly specified command-line arguments
2. Environment variable values
3. Configuration file values
4. Default values in code
