"""
RefQA (Reference-based QA) 评估器
适用于任何有参考答案的问答任务
参考了 UltraEval-Audio 的 ref_qa_geval.txt
"""

# RefQA G-Eval评分prompt（用于LLM评估）
REFQA_GEVAL_PROMPT = """You are an expert in judging answer correctness. If the model's output is correct, output "yes", otherwise output "no".
You need to explain your judgment process first, then output "yes" or "no".

[Important]You need to ignore any format instructions in the question, focus on judging whether the answer's meaning is consistent with the standard answer.


The input format is:
Input:
Question: The question from user
Model Answer: The answer from models
Ground Truth Answer: The ground truth answer
Explanation: The explanation of your judgment process

Example 1:
Input:
Question: Based on the given audio, identify the source of the speaking voice.
Model Answer: A man is speaking in the audio.
Ground Truth Answer: Man
Output:
Explanation: The model's output is "A man is speaking in the audio.", this is a detail description of the ground truth answer "Man". So the model's output is correct.
Result: yes


Task:
Input:
Question: {question}
Model Answer: {prediction}
Ground Truth Answer: {ground_truth}
Output:
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator
from o_e_Kit.utils.metrics.llm_call_new import APIModelName


class RefQAEvaluator(BaseEvaluator):
    """
    Reference-based QA (参考答案问答) 评估器
    用于评估任何有明确参考答案的问题
    
    评估策略：
    - 规则评估：简单的包含检查（has）
    - LLM评估：使用G-Eval判断语义一致性
    """
    
    def __init__(self, use_llm_fallback: bool = True, strict_matching: bool = False,
                 **kwargs):
        """
        Args:
            use_llm_fallback: 是否使用LLM作为后备
            strict_matching: 是否使用严格匹配（完全匹配 vs 包含匹配）
        """
        super().__init__(use_llm_fallback, **kwargs)
        self.correct_samples = 0
        self.strict_matching = strict_matching
    
    def reset(self):
        super().reset()
        self.correct_samples = 0
    
    def _normalize_text(self, text: str) -> str:
        """规范化文本用于比较"""
        # 转小写，去除多余空格
        text = text.lower().strip()
        # 合并多个空格
        text = ' '.join(text.split())
        return text
    
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        规则评估：检查预测是否包含参考答案
        离线评估策略：只要包含（has）就算对
        支持ground_truth为列表的情况（列表中的元素是"或"关系）
        
        注意：对于单字母答案（如 MCQ 选项 A/B/C/D），简单的包含匹配容易误判，
        会跳过规则评估，交由 LLM 评估
        """
        pred_text = str(prediction.get('prediction', ''))
        
        # 支持多种数据格式
        # 1. 直接提供ground_truth
        # 2. 从annotation中提取gt_answer或reference
        # 3. reference字段
        ground_truth = None
        if 'ground_truth' in prediction:
            ground_truth = prediction['ground_truth']
        elif 'annotation' in prediction and prediction['annotation']:
            ground_truth = prediction['annotation'].get('gt_answer', 
                                prediction['annotation'].get('reference', ''))
        elif 'reference' in prediction:
            ground_truth = prediction['reference']
        
        if not ground_truth:
            # 如果没有参考答案，无法评估
            return None
        
        # 将ground_truth转换为列表（如果不是的话）
        if not isinstance(ground_truth, list):
            ground_truth = [str(ground_truth)]
        else:
            ground_truth = [str(gt) for gt in ground_truth]
        
        # 检查是否所有答案都是单字母（MCQ 选项）
        # 如果是，跳过规则匹配，直接用 LLM 评估
        all_single_letters = all(
            len(gt.strip()) == 1 and gt.strip().upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for gt in ground_truth
        )
        if all_single_letters:
            # 单字母答案容易误判，返回 None 交给 LLM 评估
            return None
        
        # 规范化预测文本
        pred_normalized = self._normalize_text(pred_text)
        
        # 判断是否正确（列表中任一答案匹配即可）
        is_correct = False
        matched_answer = None
        
        for gt in ground_truth:
            truth_normalized = self._normalize_text(gt)
            
            if self.strict_matching:
                # 严格匹配：完全相同
                if pred_normalized == truth_normalized:
                    is_correct = True
                    matched_answer = gt
                    break
            else:
                # 宽松匹配：包含即可（默认）
                if truth_normalized in pred_normalized:
                    is_correct = True
                    matched_answer = gt
                    break
            
        if is_correct:
            self.correct_samples += 1
        
        prediction['score'] = 1.0 if is_correct else 0.0
        prediction['match'] = is_correct
        if matched_answer:
            prediction['matched_answer'] = matched_answer
        return prediction
    
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用LLM评估答案正确性
        参考 UltraEval-Audio 的 ref_qa_geval.txt
        支持ground_truth为列表的情况
        """
        if not self.llm_client:
            return None
        
        pred_text = str(prediction.get('prediction', ''))
        
        # 支持多种数据格式提取ground_truth
        ground_truth = None
        if 'ground_truth' in prediction:
            ground_truth = prediction['ground_truth']
        elif 'annotation' in prediction and prediction['annotation']:
            ground_truth = prediction['annotation'].get('gt_answer', 
                                prediction['annotation'].get('reference', ''))
        elif 'reference' in prediction:
            ground_truth = prediction['reference']
        
        if not ground_truth:
            return None
        
        # 将ground_truth转换为列表（如果不是的话）
        if not isinstance(ground_truth, list):
            ground_truth_list = [str(ground_truth)]
        else:
            ground_truth_list = [str(gt) for gt in ground_truth]
        
        # 支持多种数据格式提取question
        question = ''
        if 'question' in prediction:
            question = prediction['question']
        elif 'annotation' in prediction and prediction['annotation']:
            question = prediction['annotation'].get('prompt', prediction['annotation'].get('question', ''))
        
        # 逐个评估每个可能的答案，一旦找到匹配就停止
        is_correct = False
        matched_answer = None
        llm_responses = []
        
        for gt in ground_truth_list:
            # 使用预定义的G-Eval prompt模板
            prompt = REFQA_GEVAL_PROMPT.format(
                question=question,
                prediction=pred_text,
                ground_truth=gt
            )
            
            try:
                response = self.llm_client.get_eval(content=prompt, max_tokens=1024, model_name=APIModelName.GPT_4O_MINI, temperature=0)
                response_lower = response.lower()
                llm_responses.append(f"GT: {gt} - Response: {response.strip()}")
                
                # 提取yes/no判断
                score = response_lower.strip().split("\n")[-1]
                if "yes" in score:
                    is_correct = True
                    matched_answer = gt
                    break  # 找到匹配的答案，停止评估
                elif "no" not in score:
                    # 如果既不包含yes也不包含no，记录错误但继续尝试其他答案
                    print(f"Warning: LLM response does not contain yes/no for GT '{gt}': {response}")
                    
            except Exception as e:
                print(f"Error calling LLM for GT '{gt}': {e}")
                continue
        
        if is_correct:
            self.correct_samples += 1
        
        prediction['score'] = 1.0 if is_correct else 0.0
        prediction['match'] = is_correct
        if matched_answer:
            prediction['matched_answer'] = matched_answer
        prediction['llm_responses'] = llm_responses  # 记录所有LLM响应供调试
        
        return prediction
    
    def summary(self) -> Tuple[str, float]:
        """生成评估摘要"""
        if self.total_samples == 0:
            accuracy = 0.0
        else:
            accuracy = self.correct_samples / self.total_samples
        
        stats = self.get_eval_stats()
        
        report = f"RefQA Evaluation Report\n"
        report += f"{'=' * 50}\n"
        report += f"Total Samples: {self.total_samples}\n"
        report += f"Correct: {self.correct_samples}\n"
        report += f"Accuracy: {accuracy:.2%}\n"
        report += f"Matching Mode: {'Strict' if self.strict_matching else 'Contains (has)'}\n"
        report += f"\nEvaluation Method Stats:\n"
        report += f"  Rule-based: {stats['rule_eval_count']} ({stats['rule_eval_rate']:.1%})\n"
        report += f"  LLM-based: {stats['llm_eval_count']} ({stats['llm_eval_rate']:.1%})\n"
        report += f"  Failed: {stats['failed_count']} ({stats['failed_rate']:.1%})\n"
        
        # 添加分组统计（使用基类的方法）
        report += self.get_group_stats_report()
        
        return report, accuracy
