"""
StreamingBench评估模块
支持Real-time Visual Understanding、Omni-source Understanding、Sequential Question Answering、Proactive Output等任务的评估
"""

import re
from typing import List, Dict, Any, Optional
from collections import defaultdict


class StreamingBenchEval:
    """StreamingBench通用评估类"""
    
    def __init__(self, task_type: str = 'real'):
        """
        Args:
            task_type: 任务类型，支持 'real', 'omni', 'sqa', 'proactive'
        """
        self.task_type = task_type
        self.scored_predictions: List[Dict[str, Any]] = []
    
    def evaluate(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估预测结果
        
        Args:
            predictions: 预测结果列表，每个元素包含prediction和annotation
        
        Returns:
            评估后的结果列表
        """
        # 如果是proactive任务，使用专门的评估器
        if self.task_type == 'proactive':
            from o_e_Kit.utils.metrics.streaming_proactive_eval import StreamingProactiveEval
            proactive_evaluator = StreamingProactiveEval()
            return proactive_evaluator.evaluate(predictions)
        
        scored_predictions = []
        
        for pred in predictions:
            scored_pred = pred.copy()
            prediction_text = pred.get('prediction', '')
            annotation = pred.get('annotation', {})
            
            # 提取字段
            gt_answer = annotation.get('gt_answer', '')
            options = annotation.get('options', [])
            task_type = annotation.get('task_type', '')
            
            # 计算分数
            if options:  # 多选题
                score = self._calculate_mcq_score(prediction_text, gt_answer, options)
            else:  # 开放式问题
                score = self._calculate_open_score(prediction_text, gt_answer)
            
            # 添加评估结果
            scored_pred.update({
                'answer': prediction_text,
                'gt_answer': gt_answer,
                'task_type': task_type,
                'score': score,
                'correct': score > 0.5
            })
            
            scored_predictions.append(scored_pred)
        
        self.scored_predictions = scored_predictions
        return scored_predictions
    
    def _calculate_mcq_score(self, prediction: str, gt_answer: str, options: List[str]) -> float:
        """
        计算多选题分数
        
        Args:
            prediction: 模型预测
            gt_answer: 正确答案
            options: 选项列表
        
        Returns:
            分数 (0.0 或 1.0)
        """
        # 清理预测文本
        prediction = prediction.strip().upper()
        gt_answer = gt_answer.strip().upper()
        
        # 方法1: 直接匹配选项字母
        if prediction == gt_answer:
            return 1.0
        
        # 方法2: 提取第一个字母
        first_letter = self._extract_first_letter(prediction)
        if first_letter == gt_answer:
            return 1.0
        
        # 方法3: 匹配选项内容
        if self._match_option_content(prediction, gt_answer, options):
            return 1.0
        
        return 0.0
    
    def _calculate_open_score(self, prediction: str, gt_answer: str) -> float:
        """
        计算开放式问题分数
        """
        prediction = prediction.strip().lower()
        gt_answer = gt_answer.strip().lower()
        
        # 精确匹配
        if prediction == gt_answer:
            return 1.0
        
        # 包含匹配
        if gt_answer in prediction or prediction in gt_answer:
            return 1.0
        
        return 0.0
    
    def _extract_first_letter(self, text: str) -> str:
        """提取文本中的第一个字母"""
        match = re.search(r'[A-D]', text.upper())
        return match.group(0) if match else ''
    
    def _match_option_content(self, prediction: str, gt_answer: str, options: List[str]) -> bool:
        """匹配选项内容"""
        try:
            # 获取正确答案对应的选项内容
            answer_index = ord(gt_answer) - ord('A')
            if 0 <= answer_index < len(options):
                gt_content = options[answer_index].lower()
                # 移除选项前缀 (如 "A. ")
                gt_content = re.sub(r'^[A-D]\.\s*', '', gt_content)
                
                prediction_lower = prediction.lower()
                return gt_content in prediction_lower or prediction_lower in gt_content
        except:
            pass
        
        return False
    
    def summary(self) -> Dict[str, Any]:
        """
        计算评估总结
        
        Returns:
            包含各种指标的字典
        """
        if not self.scored_predictions:
            return {"overall_accuracy": 0.0}
        
        # 总体准确率
        total = len(self.scored_predictions)
        correct = sum(1 for pred in self.scored_predictions if pred.get('correct', False))
        overall_accuracy = correct / total if total > 0 else 0.0
        
        result = {
            "overall_accuracy": overall_accuracy,
            "total_questions": total,
            "correct_answers": correct
        }
        
        # 按任务类型分组统计
        task_stats = defaultdict(list)
        for pred in self.scored_predictions:
            task_type = pred.get('task_type', 'unknown')
            task_stats[task_type].append(pred.get('correct', False))
        
        task_accuracies = {}
        for task_type, results in task_stats.items():
            accuracy = sum(results) / len(results) if results else 0.0
            task_accuracies[task_type] = accuracy
        
        result["task_accuracies"] = task_accuracies
        
        # 按能力类型分组统计
        ability_stats = defaultdict(list)
        for pred in self.scored_predictions:
            annotation = pred.get('annotation', {})
            ability = annotation.get('required_ability', 'unknown')
            ability_stats[ability].append(pred.get('correct', False))
        
        ability_accuracies = {}
        for ability, results in ability_stats.items():
            accuracy = sum(results) / len(results) if results else 0.0
            ability_accuracies[ability] = accuracy
        
        result["ability_accuracies"] = ability_accuracies
        
        # 按视频类别分组统计
        category_stats = defaultdict(list)
        for pred in self.scored_predictions:
            annotation = pred.get('annotation', {})
            category = annotation.get('video_categories', 'unknown')
            category_stats[category].append(pred.get('correct', False))
        
        category_accuracies = {}
        for category, results in category_stats.items():
            accuracy = sum(results) / len(results) if results else 0.0
            category_accuracies[category] = accuracy
        
        result["category_accuracies"] = category_accuracies
        
        return result
    
    def print_summary(self):
        """打印评估总结"""
        summary = self.summary()
        
        print(f"\n=== StreamingBench {self.task_type.upper()} 任务评估结果 ===")
        print(f"总体准确率: {summary['overall_accuracy']:.2%}")
        print(f"总问题数: {summary['total_questions']}")
        print(f"正确答案数: {summary['correct_answers']}")
        
        if 'task_accuracies' in summary and summary['task_accuracies']:
            print(f"\n--- 按任务类型分组 ---")
            for task_type, accuracy in summary['task_accuracies'].items():
                print(f"{task_type}: {accuracy:.2%}")
        
        if 'ability_accuracies' in summary and summary['ability_accuracies']:
            print(f"\n--- 按能力类型分组 ---")
            for ability, accuracy in summary['ability_accuracies'].items():
                print(f"{ability}: {accuracy:.2%}")
        
        if 'category_accuracies' in summary and summary['category_accuracies']:
            print(f"\n--- 按视频类别分组 ---")
            for category, accuracy in summary['category_accuracies'].items():
                print(f"{category}: {accuracy:.2%}")


class StreamingBenchSQAEval(StreamingBenchEval):
    """StreamingBench Sequential Question Answering专用评估类"""
    
    def __init__(self):
        super().__init__(task_type='sqa')
    
    def evaluate(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        SQA任务的评估逻辑
        考虑序列依赖性
        """
        # 先按视频分组
        video_groups = defaultdict(list)
        for pred in predictions:
            annotation = pred.get('annotation', {})
            video_path = annotation.get('video_path', '')
            video_groups[video_path].append(pred)
        
        # 对每个视频的问题序列进行评估
        all_scored_predictions = []
        for video_path, video_preds in video_groups.items():
            # 按时间戳排序
            video_preds.sort(key=lambda x: x.get('annotation', {}).get('time_stamp', '00:00:00'))
            
            # 评估每个问题
            for pred in video_preds:
                scored_pred = super().evaluate([pred])[0]
                all_scored_predictions.append(scored_pred)
        
        self.scored_predictions = all_scored_predictions
        return all_scored_predictions 