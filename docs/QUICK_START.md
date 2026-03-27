# Quick Start Guide

Welcome to Omni-Eval Kit (o_e_Kit)! This guide will help you set up the environment and run your first evaluation.

## Step 1: Environment Setup

We strongly recommend using a Python virtual environment to avoid dependency conflicts.

1.  **Clone the Repository** (if you haven't already)

    ```bash
    git clone https://github.com/OpenBMB/OmniEvalKit.git
    cd omnievalkit
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    # Create
    python3 -m venv venv
    # Activate
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    All dependencies are listed in `requirement.txt`. Run:

    ```bash
    pip install -r requirement.txt
    ```


## Step 2: Prepare Datasets

Evaluation datasets are hosted on HuggingFace and can be downloaded with a single command:

```bash
# Download all datasets (~108GB)
python scripts/hf_download.py --output_dir ./data

# Or download specific datasets (e.g., GigaSpeech)
python scripts/hf_download.py --datasets gigaspeech_test --output_dir ./data

# List available datasets
python scripts/hf_download.py --list
```

After downloading, the `data/` directory will contain the required structure:

```
omnievalkit/
└── data/
    └── audio/
        └── asr/
            └── gigaspeech/
                ├── test.jsonl      # Annotation file
                └── test_files/     # Audio files
```

For more details, see the [Data Download Guide](./guides/DATA_DOWNLOAD.md).

## Step 3: Run Evaluation

Make sure you are in the `omnievalkit` directory, then use `torchrun` to run evaluations:

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/your/model \
    --pt_path /path/to/your/checkpoint.pt \
    --answer_path ./results \
    --model_name my_eval \
    --batchsize 4 \
    --max_sample_num 10 \
    --eval_gigaspeech_test
```

## Step 4: View Results

After evaluation, results are saved in the directory specified by `--answer_path` (defaults to `./results/` or `./answers_batch_test/`).

Directory structure:
```
<answer_path>/
└── <model_name>/
    ├── <timestamp>/
    │   └── gigaspeech_test.json  # Detailed per-sample predictions
    └── result.json               # Aggregated evaluation scores (e.g., overall WER)
```

You have successfully completed your first evaluation! For more details on architecture and extending the framework, please refer to the other documentation files.
