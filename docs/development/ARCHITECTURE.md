# Architecture and Developer Guide

This document provides a developer-focused overview of the Omni-Eval Kit's architecture. It explains the design principles, codebase structure, and how to extend the framework with new models, datasets, and evaluation metrics.

## 1. Project Philosophy

The toolkit is designed with three core principles in mind:

-   **Modularity**: Each key component (model handling, data loading, inference, evaluation) is decoupled into its own module. This makes the system easier to understand, debug, and extend.
-   **Extensibility**: Adding new functionality should be straightforward and should not require major refactoring of existing code. Clear extension points are provided for common use cases.
-   **Clarity over Complexity**: The workflows are designed to be explicit and easy to follow. For example, the main evaluation logic resides in `eval_main.py`, providing a clear entry point to the entire process.

## 2. Codebase Structure

The project is organized into the following key directories:

```
omnievalkit/
├── eval_main.py            # The main entry point for all evaluations.
├── setup.py                # Package installation config.
├── data/                   # Evaluation datasets (downloaded via scripts/hf_download.py)
├── docs/                   # Project documentation.
├── scripts/                # Helper scripts (dataset download, format conversion)
├── results/                # Evaluation results output.
├── configs/                # Global configuration files.
└── o_e_Kit/                # Core package directory.
    ├── datasets/           # Dataset implementations.
    ├── models/             # Wrappers for different models.
    │   ├── minicpm/        # MiniCPM-O series (minicpmo.py)
    │   ├── qwen/           # Qwen3-Omni series
    │   ├── gemini/         # Gemini API evaluation
    │   └── asr/            # ASR baselines (Whisper)
    ├── utils/              # Core utilities for inference, evaluation, etc.
    │   ├── args/           # Modular argument management
    │   ├── metrics/        # Evaluation metrics
    │   ├── infer.py        # Handles the distributed inference loop.
    │   ├── model_loader.py # Dynamic model loading.
    │   ├── get_args.py     # Argument parsing.
    │   └── ...
    └── configs/            # Dataset generation configs.
```

-   **`eval_main.py`**: The central orchestrator. It parses arguments, initializes the distributed environment, loads the specified model and dataset, calls the inference utility, and triggers the final evaluation.
-   **`scripts/`**: Contains helper scripts for data management (HuggingFace upload/download) and API-based evaluation.
-   **`models/`**: Contains wrappers for different models. Each wrapper can implement one or more generation methods (e.g., `generate_batch`, `generate_duplex`), which are dynamically called based on the `--generate_method` argument.
-   **`datasets/`**: Contains custom `torch.utils.data.Dataset` implementations for different evaluation benchmarks.
-   **`utils/`**: A collection of helper modules.
    -   `infer.py`: Manages the data loading (`DataLoader`), distributed sampling (`InferenceSampler`), and the core inference loop, returning raw predictions.
    -   `wer_eval.py`: Takes the raw predictions and calculates specific metrics (like WER/CER), generating a final score and a detailed report.
    -   `get_args.py`: Defines and parses all command-line arguments.

## 3. How to Extend the Framework

This is the most powerful aspect of the toolkit. Here’s how to add new components.

### How to Add a New Dataset

1.  **Create a Dataset Class**: In the `datasets/` directory, create a new Python file (e.g., `my_new_dataset.py`). Inside, define a class that inherits from `torch.utils.data.Dataset`. It must implement `__len__` and `__getitem__`. The `__getitem__` method should return a dictionary with consistent keys (e.g., `wav_path`, `question`, `gt_answers`).

2.  **Add to `get_args.py`**: Add new command-line arguments to specify the paths for your new dataset's files. Also, add a flag like `--eval_my_new_dataset` to control when it runs.

3.  **Update `eval_main.py`**:
    -   In the `load_dataset` function, add a new `elif` block to handle your new dataset's name.
    -   In the `main` function, add a new `if` block triggered by your `--eval_my_new_dataset` flag to orchestrate the evaluation flow for this dataset.

### How to Add a New Model

1.  **Create a Model Wrapper**: In the `models/` directory, create a new file for your model (e.g., `my_new_model.py`). Inside, define a class that wraps your model.
    -   The `__init__` method should handle loading the model weights.
    -   Implement **one or more generation methods** (e.g., `def my_text_generation(self, **batch)`). Each method should accept a batch dictionary and return a list of strings. The method's name should correspond to a value you'll pass to the `--generate_method` argument.

2.  **Register the Generation Method**: In `utils/infer.py`, in the `run_inference` function, add an `elif` block to call your new method based on the `generate_method` string.

3.  **Register the Model Class**: In `eval_main.py`, in the `load_model` function, add an `elif` block for your model's `--model_type` string. This block will import and instantiate your new model wrapper class.

### How to Add a New Evaluation Metric

1.  **Create an Evaluator Class**: In the `utils/` directory, you can create a new evaluation module (e.g., `vqa_eval.py`). This module should contain a class (e.g., `VQA_Eval`) with:
    -   An `evaluate` method that takes the list of predictions.
    -   A `summary` method that returns a formatted report string and a final score.

2.  **Update `evaluate_dataset`**: In `o_e_Kit/utils/wer_eval.py` (or a more general dispatch script), add a new `elif` block in the `evaluate_dataset` function. This block will be triggered by your new dataset's name and will call your new evaluator class. 