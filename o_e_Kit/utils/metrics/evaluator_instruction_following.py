"""
Instruction Following Evaluator (指令遵循评估器)
直接封装 ifeval.py 中的逻辑
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'o_e_Kit'))

from typing import Dict, Any, Optional, Tuple, List
from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator, VerboseLevel

# 直接导入 ifeval 中的函数
from o_e_Kit.utils.metrics.ifeval import (
    InputExample,
    read_prompt_list, 
    test_instruction_following_strict, 
    test_instruction_following_loose,
    read_prompt_to_response_dict,
    evaluate as ifeval_evaluate
)

class InstructionFollowingEvaluator(BaseEvaluator):
    """指令遵循评估器，继承 BaseEvaluator"""
    
    def __init__(self):
        super().__init__(use_llm_fallback=False)  # 不使用 LLM
        self.scores = []
        self.results = {}
    
    def eval(self, prediction: Dict[str, Any], verbose: VerboseLevel = VerboseLevel.NONE) -> Optional[Dict[str, Any]]:
        """
        评估单个样本的指令遵循程度
        """
        try:
            prompt = prediction['annotation']['prompt']
            pred_text = prediction['prediction']
            
            if not prompt or not pred_text:
                if verbose == VerboseLevel.INFO:
                    self.logger.info("缺少 prompt 或 prediction")
                return None
            
            # 检查是否有完整的 IFEval 数据
            ifeval_meta = prediction['annotation']['IFEvalMeta']
            instruction_id_list = ifeval_meta['instruction_id_list']
            kwargs = ifeval_meta['kwargs']
            key = ifeval_meta['key']
            
            # 如果没有完整的 IFEval 元数据，无法进行准确评估
            if not instruction_id_list or not kwargs:
                if verbose == VerboseLevel.INFO:
                    self.logger.info("缺少完整的 IFEval 元数据，无法评估")
                return None
            
            # 构造 InputExample 对象
            input_example = InputExample(
                key=key,
                instruction_id_list=instruction_id_list,
                prompt=prompt,
                kwargs=kwargs
            )
            
            # 创建单个样本的评估数据
            inputs = [input_example]
            prompt_to_response = {prompt: pred_text}
            
            # 调用 ifeval 评估
            result = ifeval_evaluate(inputs, prompt_to_response)
            
            # 提取分数
            final_score = result.get('final', 0.0)
            
            # 更新预测结果
            prediction['score'] = final_score
            prediction['ifeval_results'] = result
            prediction['eval_method'] = 'rule'
            
            # 记录分数
            self.scores.append(final_score)
            
            if verbose == VerboseLevel.INFO:
                self.logger.info(f"指令遵循评估: {final_score:.3f}")
            elif verbose == VerboseLevel.DEBUG:
                self.logger.debug(f"详细结果: {result}")
            
            return prediction
            
        except Exception as e:
            print(f"Error in InstructionFollowingEvaluator: {e}")
            return None
    
    def llm_eval(self, prediction: Dict[str, Any], verbose: VerboseLevel = VerboseLevel.NONE) -> Optional[Dict[str, Any]]:
        """
        指令遵循评估不使用 LLM
        """
        return None
    
    def summary(self) -> Tuple[str, float]:
        """
        生成评估摘要
        """
        if not self.scores:
            return "无评估数据", 0.0
        
        mean_score = sum(self.scores) / len(self.scores)
        
        report = f"Instruction Following Evaluation Summary\n"
        report += f"=" * 50 + "\n"
        report += f"Total Samples: {len(self.scores)}\n"
        report += f"Mean Score: {mean_score:.3f}\n"
        report += f"Min Score: {min(self.scores):.3f}\n"
        report += f"Max Score: {max(self.scores):.3f}\n"
        
        # 分数分布
        score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in score_ranges:
            count = sum(1 for score in self.scores if low <= score < high)
            percentage = (count / len(self.scores)) * 100
            report += f"Score {low:.1f}-{high:.1f}: {count} ({percentage:.1f}%)\n"
        
        return report, mean_score

