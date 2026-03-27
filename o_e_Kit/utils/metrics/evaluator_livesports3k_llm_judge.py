#!/usr/bin/env python3
"""
LiveSports-3K CC LLM Judge Evaluator
基于 https://github.com/showlab/livecc/blob/main/evaluation/livesports3kcc/llm_judge.py

使用 GPT-4o 作为裁判，将模型预测与 baseline (GPT-4o) 进行 AB 测试比较。
评估指标：Win Rate (胜率)
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .evaluator_base import BaseEvaluator
from .llm_call_new import APIModelName

# 默认 baseline 文件路径
DEFAULT_BASELINE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "baselines", 
    "GPT-4o.jsonl"
)


# AB Judge 提示模板 (基于 LiveCC 官方实现)
AB_JUDGE_PROMPT_TEMPLATE = """You are an expert in video commentary. 
Your task is to review two commentaries (Commentary A and Commentary B), and select the one that better aligns with the human commentary. 
You should consider the criteria:
1. Semantic Alignment: The commentary should convey the same meaning, details, and key points as the human commentary.
If the above criteria is not enough to judge, then consider:
2. Stylistic Consistency: The commentary should maintain a tone, word choice, and structure similar to the human commentary.

---Commentary A---
{a_pred}
----------

---Commentary B---
{b_pred}
----------

---Human Commentary---
{gt_asr}
----------

Your response should be "Commentary A is better aligned with the human commentary" or "Commentary B is better aligned with the human commentary"."""


@dataclass
class JudgeResult:
    """判断结果"""
    video_event_id: str
    ab_winner: str  # 第一轮 (model=A, baseline=B) 的胜者
    ba_winner: str  # 第二轮 (baseline=A, model=B) 的胜者
    score: float    # 最终得分 (0, 0.5, 1)


class LiveSports3KLLMJudgeEvaluator(BaseEvaluator):
    """
    LiveSports-3K CC LLM Judge 评估器
    
    使用 GPT-4o 作为裁判，通过 AB 测试比较模型预测与 baseline。
    为了消除位置偏差，每对样本进行两轮评估（AB 和 BA）。
    
    评估流程：
    1. 加载 baseline 预测 (GPT-4o 的预测结果)
    2. 对每个样本，使用 LLM 进行 AB 对比
    3. 计算 Win Rate
    
    继承自 BaseEvaluator，实现 eval, llm_eval, summary 抽象方法
    """
    
    def __init__(
        self,
        baseline_id: str = "GPT-4o",
        baseline_predictions: Optional[Dict[str, str]] = None,
        baseline_jsonl_path: Optional[str] = None,
        model_id: str = "Model",
        use_llm_fallback: bool = True,
        max_workers: int = 8,
        group_by_fields: List[str] = None
    ):
        """
        初始化评估器
        
        Args:
            baseline_id: Baseline 模型标识
            baseline_predictions: Baseline 预测字典 {video_event_id: prediction}
                                 如果为 None，则从 baseline_jsonl_path 加载
            baseline_jsonl_path: Baseline JSONL 文件路径
                                如果为 None，使用默认路径 (GPT-4o.jsonl)
            model_id: 当前模型标识
            use_llm_fallback: 是否使用 LLM (这里始终需要 LLM)
            max_workers: 并行评估的线程数
            group_by_fields: 分组统计字段
        """
        # 对于 LLM Judge，始终需要使用 LLM
        super().__init__(
            use_llm_fallback=True,  # 强制启用 LLM
            max_workers=max_workers,
            group_by_fields=group_by_fields or ['class']  # 默认按运动类别分组
        )
        
        self.baseline_id = baseline_id
        self.model_id = model_id
        
        # 加载 baseline 预测
        if baseline_predictions:
            self.baseline_predictions = baseline_predictions
        else:
            self.baseline_predictions = self._load_baseline(
                baseline_jsonl_path or DEFAULT_BASELINE_PATH
            )
        
        # 评估结果存储
        self.judge_results: List[JudgeResult] = []
        self.win_count = 0
        self.total_rounds = 0
    
    def _load_baseline(self, jsonl_path: str) -> Dict[str, str]:
        """
        从 JSONL 文件加载 baseline 预测
        
        文件格式:
        {"video_id": "xxx", "event_id": 10, "begin": 57.812, "end": 72.342, "pred": "..."}
        
        Args:
            jsonl_path: JSONL 文件路径
            
        Returns:
            Dict[video_event_id, prediction]
        """
        baseline_preds = {}
        
        if not os.path.exists(jsonl_path):
            print(f"⚠️ Baseline 文件不存在: {jsonl_path}")
            return baseline_preds
        
        print(f"📂 加载 baseline 预测: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        video_id = data.get('video_id', '')
                        event_id = data.get('event_id', 0)
                        pred = data.get('pred', '')
                        
                        video_event_id = f"{video_id}_{event_id}"
                        baseline_preds[video_event_id] = pred
                    except json.JSONDecodeError:
                        continue
        
        print(f"✅ 加载了 {len(baseline_preds)} 条 baseline 预测")
        return baseline_preds
        
    def reset(self):
        """重置评估统计"""
        super().reset()
        self.judge_results = []
        self.win_count = 0
        self.total_rounds = 0
    
    def _judge_ab(
        self, 
        a_id: str, 
        a_pred: str, 
        b_id: str, 
        b_pred: str, 
        gt_asr: str
    ) -> str:
        """
        使用 LLM 判断 A 和 B 哪个更接近人类解说
        
        Args:
            a_id: A 的模型标识
            a_pred: A 的预测文本
            b_id: B 的模型标识
            b_pred: B 的预测文本
            gt_asr: 人类解说 (Ground Truth ASR)
            
        Returns:
            胜者的模型标识，或 "tie"
        """
        prompt = AB_JUDGE_PROMPT_TEMPLATE.format(
            a_pred=a_pred,
            b_pred=b_pred,
            gt_asr=gt_asr
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_client.get_eval(
                    content=prompt,
                    max_tokens=100,
                    model_name=APIModelName.GPT_4O,
                    temperature=0
                )
                
                # 解析响应
                if "Commentary A" in response:
                    return a_id
                elif "Commentary B" in response:
                    return b_id
                else:
                    return "tie"
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"LLM 调用失败，重试中... ({attempt + 1}/{max_retries}): {e}")
                else:
                    print(f"LLM 调用最终失败: {e}")
                    return "error"
        
        return "tie"
    
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        规则评估 - 对于 LLM Judge 任务，规则评估始终返回 None
        因为这个任务必须使用 LLM 进行评估
        
        Args:
            prediction: 预测字典
            
        Returns:
            None (始终需要 LLM 评估)
        """
        # LLM Judge 任务没有规则评估，始终需要 LLM
        return None
    
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 进行 AB 对比评估
        
        对单个样本进行 AB 和 BA 两轮判断，消除位置偏差
        
        Args:
            prediction: 预测字典，包含:
                - prediction: 模型预测文本
                - annotation: 标注信息
                    - video_id: 视频 ID
                    - event_id: 事件 ID
                    - event_asr_text: 人类解说 (作为 GT)
                    
        Returns:
            带分数的预测结果字典
        """
        result = prediction.copy()
        annotation = prediction.get('annotation', {})
        
        # 构建 video_event_id
        video_id = annotation.get('video_id', '')
        event_id = annotation.get('event_id', 0)
        video_event_id = f"{video_id}_{event_id}"
        
        # 获取模型预测
        model_pred = prediction.get('prediction', '')
        
        # 获取人类解说 (GT)
        gt_asr = annotation.get('event_asr_text', '')
        
        # 获取 baseline 预测
        if video_event_id in self.baseline_predictions:
            baseline_pred = self.baseline_predictions[video_event_id]
        else:
            # 没有 baseline 时，跳过该样本
            print(f"⚠️ 未找到 baseline 预测: {video_event_id}，跳过评估")
            result['score'] = 0.0
            result['eval_failed'] = True
            result['extract_fail'] = 1
            result['skip_reason'] = 'no_baseline'
            return result
        
        if not model_pred or not gt_asr:
            result['score'] = 0.0
            result['eval_failed'] = True
            result['extract_fail'] = 1
            return result
        
        # 第一轮: model=A, baseline=B
        ab_winner = self._judge_ab(
            self.model_id, model_pred,
            self.baseline_id, baseline_pred,
            gt_asr
        )
        
        # 第二轮: baseline=A, model=B  
        ba_winner = self._judge_ab(
            self.baseline_id, baseline_pred,
            self.model_id, model_pred,
            gt_asr
        )
        
        # 计算得分 (0, 0.5, 1)
        score = 0.0
        if ab_winner == self.model_id:
            score += 0.5
        if ba_winner == self.model_id:
            score += 0.5
        
        # 创建 JudgeResult
        judge_result = JudgeResult(
            video_event_id=video_event_id,
            ab_winner=ab_winner,
            ba_winner=ba_winner,
            score=score
        )
        
        # 线程安全地添加结果
        with self._lock:
            self.judge_results.append(judge_result)
            # 更新胜场统计
            if ab_winner == self.model_id:
                self.win_count += 1
            if ba_winner == self.model_id:
                self.win_count += 1
            self.total_rounds += 2
        
        # 更新结果
        result['score'] = score
        result['match'] = score >= 0.5
        result['judge_result'] = {
            'video_event_id': video_event_id,
            'ab_winner': ab_winner,
            'ba_winner': ba_winner,
            'score': score
        }
        result['extract_fail'] = 0
        
        return result
    
    def summary(self) -> Tuple[str, float]:
        """
        生成评估报告
        
        Returns:
            Tuple[报告文本, 最终分数 (Win Rate)]
        """
        if not self.judge_results:
            return "No evaluation results.", 0.0
        
        # 计算 Win Rate
        win_rate = (self.win_count / self.total_rounds * 100) if self.total_rounds > 0 else 0
        
        # 详细统计
        ab_wins = sum(1 for r in self.judge_results if r.ab_winner == self.model_id)
        ba_wins = sum(1 for r in self.judge_results if r.ba_winner == self.model_id)
        ab_baseline_wins = sum(1 for r in self.judge_results if r.ab_winner == self.baseline_id)
        ba_baseline_wins = sum(1 for r in self.judge_results if r.ba_winner == self.baseline_id)
        ties = sum(1 for r in self.judge_results if r.ab_winner == "tie") + \
               sum(1 for r in self.judge_results if r.ba_winner == "tie")
        errors = sum(1 for r in self.judge_results if r.ab_winner == "error") + \
                 sum(1 for r in self.judge_results if r.ba_winner == "error")
        
        # 评估方法统计
        stats = self.get_eval_stats()
        
        # 生成报告
        report = f"""
{'='*60}
LiveSports-3K CC LLM Judge Evaluation Report
{'='*60}

📊 Overall Results:
   Model: {self.model_id}
   Baseline: {self.baseline_id}
   Total Samples: {len(self.judge_results)}
   Total Rounds: {self.total_rounds} (2 rounds per sample)

🏆 Win Rate: {win_rate:.2f}%
   Model Wins: {self.win_count} / {self.total_rounds}
   Baseline Wins: {ab_baseline_wins + ba_baseline_wins} / {self.total_rounds}

📈 Detailed Breakdown:
   Round 1 (Model=A, Baseline=B):
     - Model Wins: {ab_wins}
     - Baseline Wins: {ab_baseline_wins}
   
   Round 2 (Baseline=A, Model=B):
     - Model Wins: {ba_wins}
     - Baseline Wins: {ba_baseline_wins}
   
   Ties: {ties}
   Errors: {errors}

📋 Evaluation Stats:
   Rule Evaluations: {stats['rule_eval_count']} ({stats['rule_eval_rate']:.1%})
   LLM Evaluations: {stats['llm_eval_count']} ({stats['llm_eval_rate']:.1%})
   Failed: {stats['failed_count']} ({stats['failed_rate']:.1%})
"""
        # 添加分组统计
        group_report = self.get_group_stats_report()
        if group_report:
            report += group_report
        
        report += f"\n{'='*60}\n"
        
        return report, win_rate
    
    def save_results(self, output_path: str):
        """保存详细评估结果"""
        results_data = {
            'model_id': self.model_id,
            'baseline_id': self.baseline_id,
            'total_samples': len(self.judge_results),
            'win_rate': (self.win_count / self.total_rounds * 100) if self.total_rounds > 0 else 0,
            'judge_results': [
                {
                    'video_event_id': r.video_event_id,
                    'ab_winner': r.ab_winner,
                    'ba_winner': r.ba_winner,
                    'score': r.score
                }
                for r in self.judge_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 评估结果已保存到: {output_path}")
