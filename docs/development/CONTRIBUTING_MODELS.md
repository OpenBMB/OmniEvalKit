# How to Contribute a New Model

This guide explains how to integrate a new model into the Omni-Eval Kit framework.

## Design Principles

The framework is built around **flexibility**:
- **Dynamic dispatch**: Different generation methods are invoked dynamically via the `--generate_method` argument
- **Multiple interfaces**: Supports batch, streaming, duplex, and other inference modes
- **Modular management**: Model loading logic is separated from inference logic

## Core Steps

### 1. Register Model Parameters

First, add your model type to the choices in `o_e_Kit/utils/args/model_args.py` (if needed).

### 2. Create the Model Wrapper File

Create a new directory and file under `o_e_Kit/models/`:
```
o_e_Kit/models/
└── my_model/
    ├── __init__.py
    └── my_model.py
```

### 3. Implement the Model Wrapper Class

Define the wrapper class with initialization and generation methods:

```python
class MyModel:
    def __init__(self, model_path, device, **kwargs):
        """
        Initialize the model.

        Args:
            model_path: Path to model weights
            device: Target device (cuda/cpu)
            **kwargs: Additional model configuration
        """
        self.model = self.load_model(model_path)
        self.model.to(device)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
```

### 4. Implement Generation Methods

Implement one or more generation methods. The method name should correspond to the `--generate_method` argument value:

```python
def generate_batch(self, **batch):
    """
    Batch generation method.

    Args:
        **batch: Dictionary containing input data, which may include:
            - wav_path: List of audio file paths
            - question: List of questions
            - images: List of image data

    Returns:
        list[str]: List of model output strings
    """
    wav_paths = batch.get('wav_path', [])
    questions = batch.get('question', [])

    outputs = []
    for wav_path, question in zip(wav_paths, questions):
        audio = self.load_audio(wav_path)
        response = self.model.generate(audio, question)
        outputs.append(self.tokenizer.decode(response))

    return outputs
```

### 5. Register the Model Loader

Add the model loading logic in `o_e_Kit/utils/model_loader.py`:

```python
def load_model(args):
    """Dynamically load model based on type."""
    model_type = args.model_type

    if model_type == "minicpmo":
        from o_e_Kit.models.minicpm.minicpmo import MiniCPM_o
        return MiniCPM_o(args.model_path, args.pt_path, args.device, args.config_path)

    elif model_type == "my_model":  # Add your model
        from o_e_Kit.models.my_model.my_model import MyModel
        return MyModel(args.model_path, args.device, **vars(args))

    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 6. Register the Inference Method

In the `run_inference` function in `o_e_Kit/utils/infer.py`, ensure your generation method can be invoked:

```python
def run_inference(model, dataloader, args):
    # ...
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if args.generate_method == "batch":
                outputs = model.generate_batch(**batch)
            elif args.generate_method == "chat":
                outputs = model.generate_chat(**batch)
            elif args.generate_method == "my_custom_method":
                outputs = model.my_custom_method(**batch)
            else:
                raise ValueError(f"Unknown generate method: {args.generate_method}")
```

### 7. Run Evaluation

Now you can run evaluation with your new model:

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type my_model \
    --model_path /path/to/model \
    --generate_method batch \
    --eval_gigaspeech_test
```

## Best Practices

1. **Error handling**: Add appropriate error handling in generation methods
2. **Batch optimization**: Leverage batching to maximize inference throughput
3. **Memory management**: Clean up unnecessary intermediate variables
4. **Logging**: Add appropriate logging for debugging

## Example: Integrating an Existing Model

Using MiniCPM-O as an example of the full integration flow:

1. Model code is organized under `o_e_Kit/models/minicpm/`
2. A unified evaluation wrapper is implemented in `minicpmo.py`
3. Multiple generation methods are supported (`generate_batch`, `generate_chat`, `generate`, etc.)
4. Model and generation method are flexibly selected via command-line arguments
