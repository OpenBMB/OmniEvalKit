"""
Bargin (打断) 评估器
基于语音对话 token 序列统计 bargin 率：
- 当某个 <unit> 同时包含 <|speak|> 与 <|tts_bos|>，且其后在下一个 speak 之前出现连续 listen（listen_count > 1）时，计为一次 bargin
"""

import os
import re
from typing import Dict, Any, Optional, Tuple, List


UNIT_RE = re.compile(r"<unit>(.*?)</unit>", re.DOTALL)
TOKEN_RE = re.compile(r"<\|([a-z_]+)\|>")

def extract_units(sequence: str) -> List[str]:
    return UNIT_RE.findall(sequence or "")

def last_special_token(unit_text: str) -> Optional[str]:
    tokens = TOKEN_RE.findall(unit_text)
    return tokens[-1] if tokens else None

def analyze_sequence(sequence: str) -> Tuple[int, int, List[str]]:
    """
    Returns (speak_attempts, bargins, bargin_content_list) for a single sequence.
    Bargin 定义参考 tools/get_bargin_rate.py：listen_count > 1 视为 bargin。
    """
    units = extract_units(sequence)
    if not units:
        return 0, 0, []

    last_tokens = [last_special_token(u) for u in units]
    speak_attempts = 0
    bargins = 0
    bargin_content: List[str] = []

    i = 0
    n = len(units)
    while i < n:
        u = units[i]
        has_speak = "<|speak|>" in u
        has_tts_bos = "<|tts_bos|>" in u
        if has_speak and has_tts_bos:
            speak_attempts += 1
            j = i + 1
            listen_count = 0
            merged_u = u
            while j < n:
                # 到下一个 speak 停止；若紧邻即为另一个 speak，合并一次内容后继续看
                if "<|speak|>" in units[j]:
                    if j == i + 1:
                        merged_u += units[j]
                        j += 1
                        continue
                    break
                if last_tokens[j] == "listen":
                    listen_count += 1
                    j += 1
                    continue
                j += 1
            if listen_count > 1:
                bargin_content.append(merged_u.replace("<|audio|>", ""))
                bargins += 1
        i += 1

    return speak_attempts, bargins, bargin_content

# for test
# import sys

from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator

class BarginEvaluator(BaseEvaluator):
    """
    统计对话序列中的 bargin 率。
    - 每个样本的 `prediction` 字段或 `sequence` 字段应包含完整序列字符串。
    - 本评估器不使用 LLM 后备。
    """

    def __init__(self, use_llm_fallback: bool = False):
        super().__init__(use_llm_fallback)
        self.total_speak_attempts = 0
        self.total_bargins = 0

    def reset(self):
        super().reset()
        self.total_speak_attempts = 0
        self.total_bargins = 0

    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        规则评估：对单条样本的 sequence 统计 bargin 指标。
        为避免触发 LLM 后备与 failed 标记，这里固定返回 score=1.0（表示规则评估成功完成），
        实际指标放在扩展字段中。
        """
        # 尝试多个可能字段承载 sequence
        sequence = (
            str(prediction.get("sequence", "")) 
            if "sequence" in prediction 
            else str(prediction.get("prediction", ""))  # 兼容直接把序列放在 prediction 的情况
        )

        sa, b, bcontent = analyze_sequence(sequence)
        # 累计全局指标
        self.total_speak_attempts += sa
        self.total_bargins += b
        self.total_samples += 1
        self.rule_eval_count += 1

        # 写回样本字段
        prediction["speak_attempts"] = sa
        prediction["bargins"] = b
        prediction["bargin_rate_sample"] = (float(b) / float(sa)) if sa > 0 else 0.0
        prediction["has_bargin"] = (b > 0)
        prediction["bargin_content"] = bcontent

        # 固定为成功（不触发 LLM / failed），但不使用二分类“正确率”语义
        prediction["score"] = 1.0
        prediction["match"] = None
        return prediction

    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 不使用 LLM 评估
        return None

    def summary(self) -> Tuple[str, float]:
        """
        汇总输出整体 bargin 率（总bargins / 总speak_attempts）。
        返回 (report_string, final_score)，其中 final_score 为整体 bargin rate。
        """
        if self.total_speak_attempts == 0:
            bargin_rate = 0.0
        else:
            bargin_rate = self.total_bargins / self.total_speak_attempts

        stats = self.get_eval_stats()

        report = f"Bargin Evaluation Report\n"
        report += f"{'=' * 50}\n"
        report += f"Total Samples: {self.total_samples}\n"
        report += f"Speak Attempts: {self.total_speak_attempts}\n"
        report += f"Bargins: {self.total_bargins}\n"
        report += f"Bargin Rate: {bargin_rate:.6f}\n"
        report += f"\nEvaluation Method Stats:\n"
        report += f"  Rule-based: {stats['rule_eval_count']} ({stats['rule_eval_rate']:.1%})\n"
        report += f"  LLM-based: {stats['llm_eval_count']} ({stats['llm_eval_rate']:.1%})\n"
        report += f"  Failed: {stats['failed_count']} ({stats['failed_rate']:.1%})\n"

        return report, bargin_rate


if __name__ == "__main__":
    import json
    import sys
    import glob

    # 处理指定文件夹下所有符合条件的 JSON 文件
    folder_path = "./results/example"
    pattern = os.path.join(folder_path, "voicebench_*.json")
    
    json_files = glob.glob(pattern)
    
    # 过滤掉带有额外后缀的文件（如 voicebench_commoneval_openqa.json）
    # 只保留 voicebench_xxx.json 格式的文件
    filtered_files = []
    for f in json_files:
        basename = os.path.basename(f)
        # 检查文件名格式：voicebench_类别.json（不含额外的下划线部分）
        parts = basename.replace('.json', '').split('_')
        if len(parts) == 2:  # 只有 voicebench 和 类别 两部分
            filtered_files.append(f)
    
    json_files = filtered_files
    
    if not json_files:
        print(f"在 {folder_path} 中未找到符合条件的文件（voicebench_类别.json 格式）")
        sys.exit(1)
    
    print(f"找到 {len(json_files)} 个待处理文件")
    print("=" * 60)
    
    for json_file in sorted(json_files):
        print(f"\n处理文件: {os.path.basename(json_file)}")
        print("-" * 60)
        
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # 创建新的评估器实例（每个文件独立统计）
            evaluator = BarginEvaluator()
            
            # 处理每个预测项
            predictions = data.get('predictions', [])
            for item in predictions:
                evaluator.eval(item)
            
            # 输出当前文件的统计结果
            report, bargin_rate = evaluator.summary()
            print(report)
            
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("所有文件处理完成")