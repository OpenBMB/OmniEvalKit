# Usage Guide

This guide explains how to run evaluations using the provided scripts and how to interpret the results.

## 1. Running an Evaluation

Use `torchrun` to call `eval_main.py` for running evaluations.

### Quick Start

1.  **Navigate to the Project Root**:

    ```bash
    cd /path/to/your/project/omnievalkit
    ```

2.  **Run the Evaluation**:

    ```bash
    torchrun --nproc_per_node=1 eval_main.py \
        --model_path /path/to/your/model \
        --pt_path /path/to/your/checkpoint.pt \
        --answer_path "./results/" \
        --model_name "my_model" \
        --model_type "minicpmo" \
        --max_sample_num 10 \
        --batchsize 2 \
        --eval_gigaspeech_test
    ```

### Multi-GPU Evaluation

Control the number of GPUs via `--nproc_per_node`:

```bash
torchrun --nproc_per_node=4 --master_port=29500 eval_main.py \
    --model_path /path/to/your/model \
    --model_type minicpmo \
    --pt_path /path/to/your/checkpoint.pt \
    --answer_path ./results \
    --model_name my_model \
    --batchsize 4 \
    --eval_gigaspeech_test
```

To see all available arguments and their descriptions, run:

```bash
python eval_main.py --help
```

## 2. Understanding the Output

After an evaluation run is complete, the results are saved in the directory specified by the `--answer_path` argument. The structure is as follows:

```
<answer_path>/
└── <model_name>/
    ├── <timestamp>/
    │   └── <dataset_name>.json       # Raw predictions and per-sample details
    └── result.json                   # Final, aggregated evaluation scores
```

-   **`<dataset_name>.json`**: This file contains a list of detailed predictions for each sample, including the ground truth, the model's output, and any per-sample scores (like WER details).
-   **`result.json`**: This file contains the final, high-level evaluation metric for the entire dataset (e.g., the overall WER score).

You can inspect these files to analyze your model's performance in detail.
