"""
OV-AVEL (Open-Vocabulary Audio-Visual Event Localization) 评估器

参考论文: https://arxiv.org/pdf/2411.11278
"Towards Open-Vocabulary Audio-Visual Event Localization"

================================================================================
任务描述
================================================================================
OV-AVEL 任务要求模型在观看视频的过程中，实时检测并定位音视频事件：
- 当事件开始时，输出 "[event_name] begin"
- 当事件结束时，输出 "[event_name] end"

输入：
- 10秒视频（按1秒/帧采样 → 10帧）
- 对应的音频

输出：
- 模型在观看过程中主动输出事件开始/结束的文本

================================================================================
评测指标 (来自论文 Section 4.2)
================================================================================

1. Accuracy (帧级准确率)
   - 将10秒视频分成10个1秒片段
   - 对每个片段，判断模型预测的事件是否与 Ground Truth 匹配
   - 公式: Accuracy = (正确预测的片段数) / (总片段数)

2. Segment-level F1 (片段级 F1)
   - 对每个连续的事件片段进行评估
   - 如果预测的片段与 GT 片段的 IoU > 阈值，则认为匹配成功

3. Event-level F1 (事件级 F1)
   - 对整个事件是否被检测到进行评估

================================================================================
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re

from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator


class OVAVELEvaluator(BaseEvaluator):
    """
    OV-AVEL 事件定位评估器
    
    评估模型在音视频事件定位任务上的表现，计算：
    - Accuracy (帧级准确率)
    - Segment-level F1 (片段级 F1 分数)
    - Event-level F1 (事件级 F1 分数)
    """
    
    # 默认分组字段
    DEFAULT_GROUP_FIELDS = ['event_category', 'cls_type', 'split']
    
    def __init__(
        self,
        fps: float = 1.0,
        iou_threshold: float = 0.5,
        time_tolerance: float = 0.5,
        group_by_fields: List[str] = None,
        **kwargs
    ):
        """
        初始化 OV-AVEL 评估器
        
        Args:
            fps: 帧率 (默认 1 fps，即10秒 → 10帧)
            iou_threshold: 片段匹配的 IoU 阈值 (默认 0.5)
            time_tolerance: 时间容忍窗口，单位秒 (默认 0.5)
        """
        self.fps = fps
        self.iou_threshold = iou_threshold
        self.time_tolerance = time_tolerance
        
        # 分组字段
        if group_by_fields is None:
            group_by_fields = self.DEFAULT_GROUP_FIELDS
        
        super().__init__(group_by_fields=group_by_fields, **kwargs)
        
        # 评估统计
        self.total_frames = 0
        self.correct_frames = 0
        self.segment_tp = 0
        self.segment_fp = 0
        self.segment_fn = 0
        self.event_tp = 0
        self.event_fp = 0
        self.event_fn = 0
        
        # 分组统计
        self.group_stats = defaultdict(lambda: {
            'total_frames': 0,
            'correct_frames': 0,
            'segment_tp': 0,
            'segment_fp': 0,
            'segment_fn': 0,
            'samples': 0
        })
    
    def reset(self):
        """重置评估统计"""
        super().reset()
        self.total_frames = 0
        self.correct_frames = 0
        self.segment_tp = 0
        self.segment_fp = 0
        self.segment_fn = 0
        self.event_tp = 0
        self.event_fp = 0
        self.event_fn = 0
        self.group_stats = defaultdict(lambda: {
            'total_frames': 0,
            'correct_frames': 0,
            'segment_tp': 0,
            'segment_fp': 0,
            'segment_fn': 0,
            'samples': 0
        })
    
    def parse_model_output(self, output_text: str, duration: float = 10.0) -> List[int]:
        """
        解析模型输出，构建帧级预测标签
        
        从 duplex 模型输出中提取事件开始/结束信息，构建帧级标签
        
        Args:
            output_text: 模型输出文本，格式如 "event begin." 或 "event end."
            duration: 视频时长
        
        Returns:
            帧级标签列表，1 表示事件发生，0 表示背景
        """
        num_frames = int(duration * self.fps)
        pred_labels = [0] * num_frames
        
        # 解析 begin/end 模式
        # 格式: "[event_name] begin." 或 "[event_name] end."
        text_lower = output_text.lower().strip()
        
        # 检测是否有 begin
        if 'begin' in text_lower:
            # 事件开始，后续帧都标记为 1
            # 根据输出时间点确定开始帧（如果没有时间信息，假设从开始）
            for i in range(num_frames):
                pred_labels[i] = 1
        
        # 如果有 end，需要更精细的处理
        if 'end' in text_lower and 'begin' not in text_lower:
            # 只有 end，说明事件在这之前发生
            pred_labels = [0] * num_frames
        
        return pred_labels
    
    def parse_duplex_output(self, ai_turns: List[Dict], duration: float = 10.0) -> List[int]:
        """
        解析 duplex 推理输出的 ai_turns，构建帧级预测标签
        
        Args:
            ai_turns: duplex 输出的 ai_turns 列表，每个包含 start, end, text
            duration: 视频时长
        
        Returns:
            帧级标签列表，1 表示事件发生，0 表示背景
        """
        num_frames = int(duration * self.fps)
        pred_labels = [0] * num_frames
        
        if not ai_turns:
            return pred_labels
        
        # 追踪事件状态
        event_active = False
        event_start_frame = 0
        
        for turn in ai_turns:
            # 支持两种格式: 'text' (ai_turns) 或 'sentence' (prediction_ctc_data)
            text = turn.get('text', turn.get('sentence', '')).lower()
            start_time = turn.get('start', 0)
            
            # 计算帧索引
            frame_idx = min(int(start_time * self.fps), num_frames - 1)
            
            if 'begin' in text:
                event_active = True
                event_start_frame = frame_idx
            elif 'end' in text:
                if event_active:
                    # 标记从开始到结束的帧
                    for i in range(event_start_frame, min(frame_idx + 1, num_frames)):
                        pred_labels[i] = 1
                event_active = False
        
        # 如果事件开始但未结束，标记到视频结尾
        if event_active:
            for i in range(event_start_frame, num_frames):
                pred_labels[i] = 1
        
        return pred_labels
    
    def compute_accuracy(
        self,
        gt_labels: List[int],
        pred_labels: List[int]
    ) -> float:
        """
        计算帧级准确率
        
        Args:
            gt_labels: GT 帧级标签 (0/1)
            pred_labels: 预测帧级标签 (0/1)
        
        Returns:
            准确率 (0.0 - 1.0)
        """
        if not gt_labels or len(gt_labels) != len(pred_labels):
            return 0.0
        
        correct = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
        return correct / len(gt_labels)
    
    def _extract_segments(self, labels: List[int]) -> List[Tuple[int, int]]:
        """
        从帧级标签提取连续事件片段
        
        Args:
            labels: 帧级标签列表 (0/1)
        
        Returns:
            片段列表 [(start_frame, end_frame), ...]
        """
        segments = []
        start = None
        
        for i, label in enumerate(labels):
            if label == 1 and start is None:
                start = i
            elif label == 0 and start is not None:
                segments.append((start, i - 1))
                start = None
        
        # 处理到结尾的片段
        if start is not None:
            segments.append((start, len(labels) - 1))
        
        return segments
    
    def _compute_iou(
        self,
        seg1: Tuple[int, int],
        seg2: Tuple[int, int]
    ) -> float:
        """
        计算两个片段的 IoU
        
        Args:
            seg1, seg2: (start, end) 元组
        
        Returns:
            IoU 值 (0.0 - 1.0)
        """
        start1, end1 = seg1
        start2, end2 = seg2
        
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start > intersection_end:
            return 0.0
        
        intersection = intersection_end - intersection_start + 1
        union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_segment_f1(
        self,
        gt_labels: List[int],
        pred_labels: List[int]
    ) -> Tuple[float, float, float]:
        """
        计算片段级 F1 分数
        
        Args:
            gt_labels: GT 帧级标签
            pred_labels: 预测帧级标签
        
        Returns:
            (precision, recall, f1)
        """
        gt_segments = self._extract_segments(gt_labels)
        pred_segments = self._extract_segments(pred_labels)
        
        if not gt_segments and not pred_segments:
            return 1.0, 1.0, 1.0
        
        if not pred_segments:
            return 0.0, 0.0, 0.0
        
        if not gt_segments:
            return 0.0, 0.0, 0.0
        
        tp = 0
        matched_gt = set()
        
        for pred_seg in pred_segments:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_seg in enumerate(gt_segments):
                if i in matched_gt:
                    continue
                iou = self._compute_iou(pred_seg, gt_seg)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= self.iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
        
        fp = len(pred_segments) - tp
        fn = len(gt_segments) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def compute_event_f1(
        self,
        gt_labels: List[int],
        pred_labels: List[int]
    ) -> Tuple[float, float, float]:
        """
        计算事件级 F1 分数
        
        对于单事件，检测是否正确检测到事件的存在
        
        Args:
            gt_labels: GT 帧级标签
            pred_labels: 预测帧级标签
        
        Returns:
            (precision, recall, f1)
        """
        gt_has_event = any(l == 1 for l in gt_labels)
        pred_has_event = any(l == 1 for l in pred_labels)
        
        if gt_has_event and pred_has_event:
            # 都检测到事件
            return 1.0, 1.0, 1.0
        elif not gt_has_event and not pred_has_event:
            # 都没有事件
            return 1.0, 1.0, 1.0
        elif gt_has_event and not pred_has_event:
            # 漏检
            return 0.0, 0.0, 0.0
        else:
            # 误检
            return 0.0, 0.0, 0.0
    
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        评估单个样本
        
        Args:
            prediction: 包含模型预测和 GT 的字典
                - 'model_output': 模型输出文本 (字符串或 ai_turns 列表)
                - 'annotation': 包含 gt_label, event_category 等
        
        Returns:
            评估结果字典
        """
        result = prediction.copy()
        
        # 提取 GT 标签
        annotation = prediction.get('annotation', {})
        gt_labels = annotation.get('gt_label', [])
        duration = annotation.get('duration', 10.0)
        event_category = annotation.get('event_category', '')
        cls_type = annotation.get('cls_type', 'base')
        split = annotation.get('split', 'test')
        
        if not gt_labels:
            result['score'] = 0.0
            result['accuracy'] = 0.0
            result['segment_f1'] = 0.0
            result['event_f1'] = 0.0
            return result
        
        # 解析模型输出
        model_output = prediction.get('model_output', '')
        ai_turns = prediction.get('ai_turns', [])
        prediction_ctc_data = prediction.get('prediction_ctc_data', [])
        
        if prediction_ctc_data:
            # 使用 CTC 输出格式 (sentence, start, end)
            pred_labels = self.parse_duplex_output(prediction_ctc_data, duration)
        elif ai_turns:
            # 使用 duplex 输出格式
            pred_labels = self.parse_duplex_output(ai_turns, duration)
        elif isinstance(model_output, str) and model_output:
            pred_labels = self.parse_model_output(model_output, duration)
        else:
            pred_labels = [0] * len(gt_labels)
        
        # 确保长度一致
        if len(pred_labels) != len(gt_labels):
            pred_labels = pred_labels[:len(gt_labels)] + [0] * (len(gt_labels) - len(pred_labels))
        
        # 计算指标
        accuracy = self.compute_accuracy(gt_labels, pred_labels)
        seg_p, seg_r, seg_f1 = self.compute_segment_f1(gt_labels, pred_labels)
        evt_p, evt_r, evt_f1 = self.compute_event_f1(gt_labels, pred_labels)
        
        # 更新累积统计
        self.total_frames += len(gt_labels)
        self.correct_frames += sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
        
        # 累积 Segment 统计
        gt_segments = self._extract_segments(gt_labels)
        pred_segments = self._extract_segments(pred_labels)
        
        seg_tp = 0
        matched_gt = set()
        for pred_seg in pred_segments:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_seg in enumerate(gt_segments):
                if i in matched_gt:
                    continue
                iou = self._compute_iou(pred_seg, gt_seg)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            if best_iou >= self.iou_threshold:
                seg_tp += 1
                matched_gt.add(best_gt_idx)
        
        self.segment_tp += seg_tp
        self.segment_fp += len(pred_segments) - seg_tp
        self.segment_fn += len(gt_segments) - len(matched_gt)
        
        # 累积 Event 统计
        gt_has_event = any(l == 1 for l in gt_labels)
        pred_has_event = any(l == 1 for l in pred_labels)
        
        if gt_has_event and pred_has_event:
            self.event_tp += 1
        elif gt_has_event and not pred_has_event:
            self.event_fn += 1
        elif not gt_has_event and pred_has_event:
            self.event_fp += 1
        
        # 更新分组统计
        group_key = f"{event_category}|{cls_type}"
        self.group_stats[group_key]['total_frames'] += len(gt_labels)
        self.group_stats[group_key]['correct_frames'] += sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
        self.group_stats[group_key]['samples'] += 1
        
        # 返回结果
        result.update({
            'accuracy': accuracy,
            'segment_precision': seg_p,
            'segment_recall': seg_r,
            'segment_f1': seg_f1,
            'event_precision': evt_p,
            'event_recall': evt_r,
            'event_f1': evt_f1,
            'score': (accuracy + seg_f1 + evt_f1) / 3,
            'gt_labels': gt_labels,
            'pred_labels': pred_labels,
            'event_category': event_category,
            'cls_type': cls_type
        })
        
        return result
    
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 进行评估（当规则评估失败时）
        
        对于 OV-AVEL，规则评估通常足够
        """
        return None
    
    def _update_group_stats(self):
        """覆盖基类方法，OV-AVEL 使用自己的分组统计结构"""
        # OV-AVEL 在 eval() 方法中已经更新了 group_stats，这里不需要再次更新
        pass
    
    def summary(self) -> Tuple[str, float]:
        """
        生成评估摘要报告
        
        Returns:
            (report_string, final_score)
        """
        # 计算整体指标
        overall_accuracy = self.correct_frames / self.total_frames if self.total_frames > 0 else 0
        
        # 计算 Segment F1
        seg_precision = self.segment_tp / (self.segment_tp + self.segment_fp) if (self.segment_tp + self.segment_fp) > 0 else 0
        seg_recall = self.segment_tp / (self.segment_tp + self.segment_fn) if (self.segment_tp + self.segment_fn) > 0 else 0
        seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if (seg_precision + seg_recall) > 0 else 0
        
        # 计算 Event F1
        evt_precision = self.event_tp / (self.event_tp + self.event_fp) if (self.event_tp + self.event_fp) > 0 else 0
        evt_recall = self.event_tp / (self.event_tp + self.event_fn) if (self.event_tp + self.event_fn) > 0 else 0
        evt_f1 = 2 * evt_precision * evt_recall / (evt_precision + evt_recall) if (evt_precision + evt_recall) > 0 else 0
        
        # 计算 Avg (三个指标的平均)
        avg_score = (overall_accuracy + seg_f1 + evt_f1) / 3
        
        # 按类别统计
        base_stats = {'frames': 0, 'correct': 0, 'samples': 0}
        novel_stats = {'frames': 0, 'correct': 0, 'samples': 0}
        
        for group_key, stats in self.group_stats.items():
            parts = group_key.split('|')
            cls_type = parts[1] if len(parts) > 1 else 'base'
            
            if cls_type == 'open' or cls_type == 'novel':
                novel_stats['frames'] += stats['total_frames']
                novel_stats['correct'] += stats['correct_frames']
                novel_stats['samples'] += stats['samples']
            else:
                base_stats['frames'] += stats['total_frames']
                base_stats['correct'] += stats['correct_frames']
                base_stats['samples'] += stats['samples']
        
        base_acc = base_stats['correct'] / base_stats['frames'] if base_stats['frames'] > 0 else 0
        novel_acc = novel_stats['correct'] / novel_stats['frames'] if novel_stats['frames'] > 0 else 0
        
        # 构建报告
        report = f"""
==================== OV-AVEL Evaluation Report ====================

Overall Metrics:
-----------------
Total Samples:          {self.total_samples}
Total Frames:           {self.total_frames}

Acc. (Frame Accuracy):  {overall_accuracy*100:.1f}
Seg. (Segment F1):      {seg_f1*100:.1f}
Eve. (Event F1):        {evt_f1*100:.1f}
Avg.:                   {avg_score*100:.1f}

By Category Type:
-----------------
Base Categories:
  Samples:              {base_stats['samples']}
  Frames:               {base_stats['frames']}
  Accuracy:             {base_acc:.4f}

Novel/Open Categories:
  Samples:              {novel_stats['samples']}
  Frames:               {novel_stats['frames']}
  Accuracy:             {novel_acc:.4f}

===================================================================
"""
        
        final_score = avg_score
        return report, final_score
