# How to Contribute a New Evaluation Method

This guide explains how to add a new evaluation metric or a complete evaluation pipeline to the Omni-Eval Kit framework.

## Design Principles

- **Unified interface**: All evaluators inherit from `EvaluatorBase` for consistent APIs
- **Modular management**: Evaluators are organized by task type under `metrics/`
- **Flexible extension**: Supports different metric types (accuracy, WER, BLEU, etc.)
- **Standardized results**: Unified result format for easy comparison and analysis

## Core Steps

### 1. Create the Evaluator File

Create a new Python file under `o_e_Kit/utils/metrics/`, e.g., `evaluator_vqa.py`.

### 2. Implement the Evaluator Class

Your evaluator should inherit from `EvaluatorBase`:

```python
# o_e_Kit/utils/metrics/evaluator_vqa.py

from .evaluator_base import EvaluatorBase

class VQAEvaluator(EvaluatorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.correct_count = 0
        self.total_count = 0
        self.predictions = []

    def evaluate(self, predictions, references=None):
        """
        Evaluate predictions against references.

        Args:
            predictions: List of model predictions
            references: Reference answers (if not included in predictions)

        Returns:
            dict: Evaluation results
        """
        scored_predictions = []

        for item in predictions:
            gt_answers = item.get('gt_answers', [])
            model_answer = item.get('answer', '')

            is_correct = self._check_answer(model_answer, gt_answers)

            if is_correct:
                self.correct_count += 1
            self.total_count += 1

            scored_item = item.copy()
            scored_item['score'] = 1 if is_correct else 0
            scored_predictions.append(scored_item)

        self.predictions = scored_predictions
        return self.get_results()

    def _check_answer(self, prediction, references):
        pred = prediction.lower().strip()
        return any(pred == ref.lower().strip() for ref in references)

    def get_results(self):
        accuracy = (self.correct_count / self.total_count) * 100 if self.total_count > 0 else 0
        return {
            'accuracy': accuracy,
            'correct': self.correct_count,
            'total': self.total_count,
            'predictions': self.predictions
        }

    def summary(self):
        """Generate an evaluation report."""
        results = self.get_results()
        accuracy = results['accuracy']

        report = (
            f"\n{'='*50}\n"
            f"VQA Evaluation Summary\n"
            f"{'='*50}\n"
            f"Total Samples:    {self.total_count}\n"
            f"Correct:          {self.correct_count}\n"
            f"Accuracy:         {accuracy:.2f}%\n"
            f"{'='*50}\n"
        )

        return report, accuracy
```

### 3. Register the Evaluator

Register your evaluator in `o_e_Kit/utils/evaluation_runner.py` or `evaluation_runner_audio.py`:

```python
def evaluate_dataset(dataset_name, predictions, args):
    if dataset_name in ["gigaspeech_test", "wenetspeech_test_net"]:
        from o_e_Kit.utils.metrics.wer_eval import WER_Eval
        evaluator = WER_Eval(lang='en' if 'gigaspeech' in dataset_name else 'zh')
        results = evaluator.evaluate(predictions)

    elif dataset_name == "vqa_dataset":  # Add your dataset
        from o_e_Kit.utils.metrics.evaluator_vqa import VQAEvaluator
        evaluator = VQAEvaluator()
        results = evaluator.evaluate(predictions)

    report, score = evaluator.summary()
    print(report)

    return results, score
```

### 4. Configure Datasets to Use the New Evaluator

Ensure your dataset is mapped to the correct evaluator in the evaluation pipeline.

## Advanced Features

### Supporting Multiple Metrics

```python
class MultiMetricEvaluator(EvaluatorBase):
    def evaluate(self, predictions, references=None):
        results = {}
        results['accuracy'] = self._compute_accuracy(predictions)
        results['f1_score'] = self._compute_f1(predictions)
        results['bleu'] = self._compute_bleu(predictions)
        return results
```

### Supporting Custom Configuration

```python
class ConfigurableEvaluator(EvaluatorBase):
    def __init__(self, config_path=None, **kwargs):
        super().__init__(**kwargs)
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs
```

## Best Practices

1. **Inherit from the base class**: Always extend `EvaluatorBase` for interface consistency
2. **Error handling**: Add appropriate exception handling and input validation
3. **Documentation**: Provide detailed docstrings for your evaluator
4. **Testing**: Write test cases to verify evaluator correctness
5. **Performance**: Consider batching and parallelization for large datasets

## Existing Evaluators for Reference

- `wer_eval.py`: Speech recognition (WER/CER)
- `evaluator_caption.py`: Image/video captioning
- `evaluator_mqa.py`: Multiple-choice QA
- `evaluator_openqa.py`: Open-ended QA
- `evaluator_safety.py`: Safety evaluation
- `llm_call_new.py`: LLM-based evaluation
