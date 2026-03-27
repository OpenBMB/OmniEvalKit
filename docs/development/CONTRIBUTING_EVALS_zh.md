# 如何贡献一个新的评测方法

本指南详细说明了如何为 Omni-Eval Kit 框架添加一个新的评测指标或一套完整的评测流程。

## 设计原则

- **统一接口**：所有评估器继承自 `EvaluatorBase` 基类，确保接口一致性
- **模块化管理**：评估器按任务类型组织在 `metrics/` 目录下
- **灵活扩展**：支持不同类型的评估指标（准确率、WER、BLEU等）
- **结果标准化**：统一的结果格式便于比较和分析

## 核心步骤

### 1. 创建评估器文件

在 `o_e_Kit/utils/metrics/` 目录下，为您的新评测方法创建一个 Python 文件，例如 `evaluator_vqa.py`。

### 2. 实现评估器类

您的评估器应该继承自 `EvaluatorBase` 基类：

```python
# o_e_Kit/utils/metrics/evaluator_vqa.py

from .evaluator_base import EvaluatorBase

class VQAEvaluator(EvaluatorBase):
    def __init__(self, **kwargs):
        """
        初始化评估器
        
        Args:
            **kwargs: 可选配置参数
        """
        super().__init__(**kwargs)
        self.correct_count = 0
        self.total_count = 0
        self.predictions = []

    def evaluate(self, predictions, references=None):
        """
        评估预测结果
        
        Args:
            predictions: 模型预测结果列表
            references: 参考答案（如果不在predictions中）
        
        Returns:
            dict: 包含评估结果的字典
        """
        scored_predictions = []
        
        for item in predictions:
            # 获取预测和参考答案
            gt_answers = item.get('gt_answers', [])
            model_answer = item.get('answer', '')
            
            # 计算得分
            is_correct = self._check_answer(model_answer, gt_answers)
            
            if is_correct:
                self.correct_count += 1
            self.total_count += 1
            
            # 保存评分结果
            scored_item = item.copy()
            scored_item['score'] = 1 if is_correct else 0
            scored_predictions.append(scored_item)
        
        self.predictions = scored_predictions
        return self.get_results()

    def _check_answer(self, prediction, references):
        """检查答案是否正确"""
        # 标准化处理
        pred = prediction.lower().strip()
        return any(pred == ref.lower().strip() for ref in references)

    def get_results(self):
        """获取评估结果"""
        accuracy = (self.correct_count / self.total_count) * 100 if self.total_count > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': self.correct_count,
            'total': self.total_count,
            'predictions': self.predictions
        }

    def summary(self):
        """生成评估报告"""
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

### 3. 注册评估器

在 `o_e_Kit/utils/evaluation_runner.py` 或 `evaluation_runner_audio.py` 中注册您的评估器：

```python
# o_e_Kit/utils/evaluation_runner_audio.py

def evaluate_dataset(dataset_name, predictions, args):
    """根据数据集类型选择评估器"""
    
    if dataset_name in ["gigaspeech_test", "wenetspeech_test_net"]:
        from o_e_Kit.utils.metrics.wer_eval import WER_Eval
        evaluator = WER_Eval(lang='en' if 'gigaspeech' in dataset_name else 'zh')
        results = evaluator.evaluate(predictions)
        
    elif dataset_name == "vqa_dataset":  # 添加您的数据集
        from o_e_Kit.utils.metrics.evaluator_vqa import VQAEvaluator
        evaluator = VQAEvaluator()
        results = evaluator.evaluate(predictions)
        
    else:
        # 默认评估器或抛出异常
        raise ValueError(f"No evaluator for dataset: {dataset_name}")
    
    # 生成报告
    report, score = evaluator.summary()
    print(report)
    
    return results, score
```

### 4. 配置数据集使用新评估器

确保您的数据集在运行时能够调用到正确的评估器。这通常在数据集配置或评估流程中指定。

## 高级功能

### 1. 支持多种评估指标

```python
class MultiMetricEvaluator(EvaluatorBase):
    def evaluate(self, predictions, references=None):
        results = {}
        
        # 计算多个指标
        results['accuracy'] = self._compute_accuracy(predictions)
        results['f1_score'] = self._compute_f1(predictions)
        results['bleu'] = self._compute_bleu(predictions)
        
        return results
```

### 2. 支持外部评估工具

```python
class ExternalEvaluator(EvaluatorBase):
    def evaluate(self, predictions, references=None):
        # 调用外部API或工具
        import requests
        response = requests.post(
            "https://api.example.com/evaluate",
            json={"predictions": predictions}
        )
        return response.json()
```

### 3. 支持自定义评估配置

```python
class ConfigurableEvaluator(EvaluatorBase):
    def __init__(self, config_path=None, **kwargs):
        super().__init__(**kwargs)
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = kwargs
```

## 最佳实践

1. **继承基类**：始终从 `EvaluatorBase` 继承，确保接口一致性
2. **错误处理**：添加适当的异常处理和输入验证
3. **文档完善**：为评估器添加详细的文档字符串
4. **单元测试**：编写测试用例验证评估器的正确性
5. **性能优化**：对大规模数据集考虑批处理和并行化

## 现有评估器参考

项目中已实现的评估器可作为参考：
- `wer_eval.py`: 语音识别评估（WER/CER）
- `evaluator_caption.py`: 图像/视频描述评估
- `evaluator_mqa.py`: 多选题评估
- `evaluator_openqa.py`: 开放问答评估
- `evaluator_safety.py`: 安全性评估
- `llm_call.py`: 基于LLM的评估 