"""
OpenQA (Open-ended QA) 评估器
# copy from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/datasets/ds_openqa.py
"""

# 标准的OpenQA评分prompt（1-5分制）
OPEN_QA_PROMPT_5SCALE = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user's question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don't contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user's question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user's query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user's instruction and models' response:
### [Instruction]: {question}
### [Response]: {prediction}

After evaluating, please output the score only without anything else.
You don't need to provide any explanations.
"""

import re
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator
from o_e_Kit.utils.metrics.llm_call_new import APIModelName


class OpenQAEvaluator(BaseEvaluator):
    """
    Open-ended QA (开放式问答) 评估器
    特点：
    - 1-5分制独立评分（不需要参考答案）
    - 拒绝检测（用于安全性评估）
    - 支持多种评分格式提取
    """
    
    def __init__(self, use_llm_fallback: bool = True, 
                 check_refusal: bool = False):
        """
        Args:
            use_llm_fallback: 是否使用LLM作为后备
            check_refusal: 是否检测拒绝回答
        """
        super().__init__(use_llm_fallback)
        self.scores = []
    
    def reset(self):
        super().reset()
        self.scores = []
    
    def _extract_rating(self, llm_output: str) -> Optional[float]:
        """
        从LLM输出中提取评分（1-5分制）
        支持多种格式：纯数字、[[数字]]、带文本的数字
        """
        llm_output = llm_output.strip()
        
        # 1. 尝试直接转换为数字
        try:
            score = float(llm_output)
            if 1 <= score <= 5:
                return score
        except ValueError:
            pass
        
        # 2. 尝试提取[[数字]]格式
        pattern = r'\[\[(\d+\.?\d*)\]\]'
        match = re.search(pattern, llm_output)
        if match:
            score = float(match.group(1))
            if 1 <= score <= 5:
                return score
        
        # 3. 尝试提取任何数字
        numbers = re.findall(r'\d+\.?\d*', llm_output)
        if numbers:
            # 寻找1-5范围内的数字
            for num in numbers:
                num = float(num)
                if 1 <= num <= 5:
                    return num
            # 如果没有合理范围内的数字，但有数字，尝试第一个
            if numbers:
                score = float(numbers[0])
                # 限制在1-5范围内
                return max(1.0, min(5.0, score))
        
        return None
    
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        规则评估：只处理拒绝检测
        OpenQA通常需要LLM评估
        """
        
        # OpenQA通常需要LLM评估
        return None
    
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用LLM评估答案质量（1-5分制）
        不需要参考答案
        """
        if not self.llm_client:
            # 没有LLM时无法评估
            return None
        
        pred_text = str(prediction.get('prediction', ''))
        
        # 支持多种数据格式提取question
        question = ''
        if 'question' in prediction:
            question = prediction['question']
        elif 'annotation' in prediction and prediction['annotation']:
            question = prediction['annotation'].get('prompt', prediction['annotation'].get('question', ''))
        
        # 使用1-5分制prompt
        prompt = OPEN_QA_PROMPT_5SCALE.format(
            question=question,
            prediction=pred_text
        )
        
        try:
            response = self.llm_client.get_eval(content=prompt, max_tokens=10, temperature=0.0, model_name=APIModelName.GPT_4O_MINI)
            score = self._extract_rating(response)
            
            if score is not None:
                self.scores.append(score)
                prediction['score'] = score  # 原始分数（1-5）
                prediction['llm_response'] = response
                return prediction
                
        except Exception as e:
            print(f"Error calling LLM for evaluation: {e}")
        
        return None
    

    
    def summary(self) -> Tuple[str, float]:
        """生成评估摘要"""
        if len(self.scores) == 0:
            mean_score = 0.0
        else:
            mean_score = np.mean(self.scores)
        
        stats = self.get_eval_stats()
        
        report = f"OpenQA Evaluation Report\n"
        report += f"{'=' * 50}\n"
        report += f"Total Samples: {self.total_samples}\n"
        report += f"Scoring Scale: 1-5 (LLM-based)\n"
        report += f"Mean Score: {mean_score:.2f}\n"
        
        if len(self.scores) > 0:
            # 计算原始分数（1-5）的统计
            raw_scores = [(s * 4) + 1 for s in self.scores]  # 从归一化恢复到1-5
            report += f"\nScore Distribution:\n"
            report += f"  Raw (1-5): Mean={np.mean(raw_scores):.2f}, Min={np.min(raw_scores):.2f}, Max={np.max(raw_scores):.2f}\n"
        
        report += f"\nEvaluation Method Stats:\n"
        report += f"  Rule-based: {stats['rule_eval_count']} ({stats['rule_eval_rate']:.1%})\n"
        report += f"  LLM-based: {stats['llm_eval_count']} ({stats['llm_eval_rate']:.1%})\n"
        report += f"  Failed: {stats['failed_count']} ({stats['failed_rate']:.1%})\n"
        
        return report, mean_score
