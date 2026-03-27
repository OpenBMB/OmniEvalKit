"""
评估器基础框架
提供统一的评估器接口和基类
实现智能评估策略：先尝试规则评估，必要时使用LLM
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re
from collections import defaultdict
from o_e_Kit.utils.metrics.llm_call_new import ChatClient, APIModelName
from enum import Enum
from o_e_Kit.utils.logger.simple_progress import smart_progress
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class VerboseLevel(Enum):
    """详细级别枚举"""
    NONE = 0      # 无输出
    INFO = 1      # 总结性信息
    DEBUG = 2     # 关键路径信息

class BaseEvaluator(ABC):
    """
    评估器基类
    所有评估器都应该继承这个类
    
    评估策略：
    1. 首先尝试使用规则/模式匹配进行评估 (eval)
    2. 如果规则无法处理，则使用LLM进行评估 (llm_eval)
    """
    
    def __init__(self, use_llm_fallback: bool = True, max_workers: int = 16,
                 group_by_fields: List[str] = None):
        """
        初始化评估器
        
        Args:
            use_llm_fallback: 当规则评估失败时，是否使用LLM作为后备
            max_workers: 并行评估的最大线程数
            group_by_fields: 分组统计字段列表，如 ['task', 'subset_name']
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_client = None
        self.max_workers = max_workers
        self.group_by_fields = group_by_fields or []
        
        if use_llm_fallback:
            self.llm_client = ChatClient()
        
        # 用于线程安全的锁
        self._lock = threading.Lock()
        
        self.reset()
    
    def _init_group_stats(self):
        """初始化分组统计"""
        self.group_stats = {}
        for field in self.group_by_fields:
            self.group_stats[field] = defaultdict(lambda: {'count': 0, 'correct': 0})
    
    def reset(self):
        """重置评估统计"""
        self.total_samples = 0
        self.rule_eval_count = 0  # 规则评估成功的数量
        self.llm_eval_count = 0   # LLM评估的数量
        self.failed_count = 0     # 评估失败的数量
        self.scored_predictions = []
        self.task_stats = {}  # 重置 task 统计
        self._init_group_stats()  # 初始化分组统计
    
    @abstractmethod
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用规则/模式匹配进行评估
        
        Args:
            prediction: 包含预测和标签的字典
        
        Returns:
            如果能够评估，返回评分结果字典；否则返回None
        """
        raise NotImplementedError
    
    @abstractmethod
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用LLM进行评估
        
        Args:
            prediction: 包含预测和标签的字典
        
        Returns:
            如果能够评估，返回评分结果字典；否则返回None
        """
        raise NotImplementedError
    
    def evaluate_single(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个预测
        先尝试规则评估，失败则使用LLM
        
        Args:
            prediction: 包含预测和标签的字典
        
        Returns:
            包含评分结果的字典
        """
        # 首先尝试规则评估（不需要加锁，假设 eval 方法是线程安全的）
        result = self.eval(prediction)
        eval_method = None
        
        if result is not None and result['score'] == 1.0:
            eval_method = 'rule'
            result['eval_method'] = 'rule'
        else:
            # 如果规则评估失败且启用了LLM后备
            if self.use_llm_fallback and self.llm_client:
                result = self.llm_eval(prediction)
                if result is not None and result.get('extract_fail', 0) == 0:
                    eval_method = 'llm'
                    result['eval_method'] = 'llm'
                else:
                    # LLM评估也失败了
                    eval_method = 'failed'
                    if result is None:
                        result = prediction.copy()
                    result['eval_method'] = 'failed'
                    result['score'] = 0.0
                    result['eval_failed'] = True
            else:
                # 没有启用LLM后备
                eval_method = 'failed'
                if result is None:
                    result = prediction.copy()
                result['eval_method'] = 'failed'
                result['score'] = 0.0
                result['eval_failed'] = True
        
        # 使用锁来更新共享状态
        with self._lock:
            self.total_samples += 1
            
            if eval_method == 'rule':
                self.rule_eval_count += 1
            elif eval_method == 'llm':
                self.llm_eval_count += 1
            elif eval_method == 'failed':
                self.failed_count += 1
            
            # 统一处理 task 统计
            task = None
            if 'annotation' in result and 'task' in result['annotation']:
                task = result['annotation']['task']
            elif 'task' in result:
                task = result['task']
                
            if task:
                if task not in self.task_stats:
                    self.task_stats[task] = {'total': 0, 'correct': 0}
                self.task_stats[task]['total'] += 1
                if result.get('score', 0) == 1.0:
                    self.task_stats[task]['correct'] += 1
        
        return result
    
    def _evaluate_chunk(self, chunk: List[Dict[str, Any]], chunk_id: int) -> List[Dict[str, Any]]:
        """
        评估一个数据块
        """
        chunk_results = []
        # 为每个 worker 创建独立的进度条描述
        desc = f"Worker {chunk_id} evaluating"
        
        for pred in smart_progress(chunk, desc=desc):
            scored_pred = self.evaluate_single(pred)
            chunk_results.append(scored_pred)
        
        return chunk_results
    
    def evaluate(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估一批预测
        - 当 use_llm_fallback=True 时，使用多线程并行（LLM调用慢）
        - 当 use_llm_fallback=False 时，使用单线程（规则评估快）
        
        Args:
            predictions: 预测列表
        
        Returns:
            评分后的预测列表
        """
        self.reset()
        
        # 决定是否使用并行
        should_use_parallel = (
            self.use_llm_fallback and  # 启用了 LLM fallback
            self.max_workers > 1 and   # 有多个 worker
            len(predictions) >= 10     # 样本数量足够
        )
        
        # 单线程处理（规则评估或样本太少）
        if not should_use_parallel:
            desc = "Evaluating predictions (single-threaded)"
                
            for pred in smart_progress(predictions, desc=desc):
                scored_pred = self.evaluate_single(pred)
                self.scored_predictions.append(scored_pred)
            
            # 更新分组统计
            self._update_group_stats()
            return self.scored_predictions
        
        # 多线程处理（LLM评估）
        # 将数据分成n份
        n_workers = min(self.max_workers, len(predictions))
        chunk_size = len(predictions) // n_workers
        remainder = len(predictions) % n_workers
        
        chunks = []
        start = 0
        for i in range(n_workers):
            # 前remainder个worker多处理一个样本
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(predictions[start:end])
            start = end
        
        print(f"🚀 使用 {n_workers} 个线程并行评估 {len(predictions)} 个样本（LLM模式）...")
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i}: {len(chunk)} 个样本")
        
        # 使用线程池并行处理每个chunk
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有chunk的评估任务
            futures = [
                executor.submit(self._evaluate_chunk, chunk, i)
                for i, chunk in enumerate(chunks)
            ]
            
            # 收集结果
            all_results = []
            for i, future in enumerate(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    print(f"✅ Worker {i} 完成，处理了 {len(chunk_results)} 个样本")
                except Exception as e:
                    print(f"❌ Worker {i} 出错: {e}")
        
        self.scored_predictions = all_results
        print(f"\n✅ 评估完成！共评估 {len(self.scored_predictions)} 个样本")
        
        # 更新分组统计
        self._update_group_stats()
        
        return self.scored_predictions
    
    def _update_group_stats(self):
        """更新分组统计（在评估完成后调用）"""
        if not self.group_by_fields or not self.scored_predictions:
            return
        
        self._init_group_stats()
        for pred in self.scored_predictions:
            is_correct = pred.get('match', False) or (pred.get('score', 0) == 1.0)
            annotation = pred.get('annotation', {})
            
            for field in self.group_by_fields:
                # 优先从 annotation 获取，其次从顶层获取
                field_value = annotation.get(field) if annotation.get(field) is not None else pred.get(field, 'unknown')
                field_value = str(field_value) if field_value is not None else 'unknown'
                
                self.group_stats[field][field_value]['count'] += 1
                if is_correct:
                    self.group_stats[field][field_value]['correct'] += 1
    
    @abstractmethod
    def summary(self) -> Tuple[str, float]:
        """
        生成评估摘要
        
        Returns:
            (report_string, final_score)
        """
        raise NotImplementedError
    
    def get_eval_stats(self) -> Dict[str, Any]:
        """
        获取评估统计信息
        """
        return {
            'total_samples': self.total_samples,
            'rule_eval_count': self.rule_eval_count,
            'llm_eval_count': self.llm_eval_count,
            'failed_count': self.failed_count,
            'rule_eval_rate': self.rule_eval_count / self.total_samples if self.total_samples > 0 else 0,
            'llm_eval_rate': self.llm_eval_count / self.total_samples if self.total_samples > 0 else 0,
            'failed_rate': self.failed_count / self.total_samples if self.total_samples > 0 else 0,
        }
    
    def get_task_stats_report(self) -> str:
        """
        生成 task 统计报告
        
        Returns:
            格式化的 task 统计报告字符串
        """
        if not self.task_stats:
            return ""
        
        report = "\nTask-wise Accuracy:\n"
        report += f"{'-' * 50}\n"
        report += f"{'Task':<20} {'Total':<10} {'Correct':<10} {'Accuracy':<10}\n"
        report += f"{'-' * 50}\n"
        
        task_names = sorted(self.task_stats.keys())
        for task in task_names:
            stats = self.task_stats[task]
            task_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            report += f"{task:<20} {stats['total']:<10} {stats['correct']:<10} {task_accuracy:<10.2%}\n"
        
        # 显示所有样本级别的统计（与上面的 Total Samples 保持一致）
        total_correct = sum(s['correct'] for s in self.task_stats.values())
        overall_accuracy = total_correct / self.total_samples if self.total_samples > 0 else 0.0
        report += f"{'-' * 50}\n"
        report += f"{'Overall':<20} {self.total_samples:<10} {total_correct:<10} {overall_accuracy:<10.2%}\n"
        
        return report
    
    def get_group_stats_report(self) -> str:
        """
        生成分组统计报告
        
        Returns:
            格式化的分组统计报告字符串
        """
        if not self.group_by_fields or not self.group_stats:
            return ""
        
        report = "\nGroup-by Statistics:\n"
        for field in self.group_by_fields:
            if field not in self.group_stats or not self.group_stats[field]:
                continue
            
            report += f"\n  By {field}:\n"
            # 按样本数从多到少排序
            for value, stats in sorted(self.group_stats[field].items(), key=lambda x: -x[1]['count']):
                if stats['count'] > 0:
                    acc = stats['correct'] / stats['count']
                    report += f"    {value}: {acc:.2%} ({stats['correct']}/{stats['count']})\n"
        
        return report