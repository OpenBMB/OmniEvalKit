import re
from typing import List, Dict, Any, Optional
from o_e_Kit.utils.ovo_bench_utils import BACKWARD_TASKS, REAL_TIME_TASKS, FORWARD_TASKS

class MCQ_Eval:
    def __init__(self, metric: str='MCQ'):
        self.metric = metric
        self.scored_predictions: List[Dict[str, Any]] = []

    def evaluate(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """评估MCQ格式的预测结果，适配新的标准化数据格式"""
        scored_predictions = []
        
        for pred in predictions:
            scored_pred = pred.copy()
            
            # 适配新的数据格式：从推理结果中提取信息
            prediction_text = pred.get('prediction', '')  # 模型预测结果
            annotation = pred.get('annotation', {})  # 标注信息（新格式）
            
            # 从annotation中提取评估所需的字段
            gt_answer = annotation.get('gt_answer', '')
            task = annotation.get('task_type', '')  # 新格式使用task_type
            gt_index = annotation.get('gt_index', None)
            sample_id = annotation.get('sample_id', '')
            options = annotation.get('options', [])
            question = annotation.get('question', '')
            
            # 为了兼容性和调试，添加提取的字段到结果中
            scored_pred.update({
                'answer': prediction_text,  # 模型预测
                'gt_answer': gt_answer,     # 正确答案
                'task': task,               # 任务类型
                'gt_index': gt_index,       # 正确选项索引
                'sample_id': sample_id,     # 样本ID
                'options': options,         # 选项列表
                'question': question        # 问题
            })
            
            # 计算分数
            score = self._calculate_score(prediction_text, gt_answer, task, gt_index)
            scored_pred['score'] = score
            
            scored_predictions.append(scored_pred)
        
        # 保存评估结果以供summary使用
        self.scored_predictions = scored_predictions
        return scored_predictions

    def _calculate_score(self, answer: str, gt_answer: str, task: str, gt_index: Optional[int] = None) -> float:
        """根据任务类型计算分数，参考OVOBenchScore.py的逻辑"""
        if not answer:
            return 0.0
        
        # Backward和Realtime任务：多选题，应该比较选项字母
        if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
            # 提取模型回答中的选项字母
            predicted_letter = self._extract_option_letter(answer)
            if predicted_letter and gt_index is not None:
                # 将gt_index转换为字母 (0->A, 1->B, 2->C, 3->D)
                ground_truth = chr(65 + gt_index)  # 对应您提到的逻辑
                return 1.0 if predicted_letter == ground_truth else 0.0
            else:
                # 兜底：使用原始逻辑
                return 1.0 if gt_answer in answer else 0.0
        
        # REC任务：提取数字进行精确匹配
        elif task == "REC":
            return self._calculate_rec_score(answer, gt_answer)
        
        # SSR/CRR任务：检查Yes/No答案（对齐官方逻辑）
        elif task in ["SSR", "CRR"]:
            return self._calculate_ssr_crr_score(answer, gt_answer)
        
        else:
            # 默认检查答案匹配
            return 1.0 if gt_answer in answer else 0.0

    def _extract_option_letter(self, answer: str) -> str:
        """从模型回答中提取选项字母 (A, B, C, D)"""
        # 查找第一个出现的A、B、C、D字母
        match = re.search(r'[ABCD]', answer.upper())
        return match.group(0) if match else ''

    def _calculate_rec_score(self, answer: str, gt_answer: str) -> float:
        """计算REC任务的分数，参考OVOBenchScore.py的get_score_REC"""
        # 提取数字
        answer_nums = re.findall(r'\d+', answer)
        gt_nums = re.findall(r'\d+', gt_answer)
        
        if not answer_nums or not gt_nums:
            return 0.0
        
        # 检查是否有匹配的数字
        answer_str = "".join(answer_nums)
        gt_str = "".join(gt_nums)
        
        return 1.0 if answer_str == gt_str else 0.0

    def _calculate_ssr_crr_score(self, answer: str, gt_answer: str) -> float:
        """计算SSR和CRR任务的分数，参考OVOBenchScore.py的get_score_SSR_CRR"""
        if not answer:
            return 0.0
            
        answer_clean = answer.strip().lower()
        gt_clean = gt_answer.strip().lower()
        
        if answer_clean == "n":
            answer_clean = "no"
        if answer_clean == "y":
            answer_clean = "yes"
            
        # 检查gt_answer是否包含在answer中（都转为小写进行比较）
        return 1.0 if gt_clean in answer_clean else 0.0

    def summary(self) -> tuple[str, float]:
        """生成评估报告和最终分数，对齐OVO-Bench原项目的评分逻辑"""
        if not hasattr(self, 'scored_predictions') or not self.scored_predictions:
            raise ValueError("Please call evaluate() first")
        
        # 按任务类型分组统计
        backward_tasks = BACKWARD_TASKS
        realtime_tasks = REAL_TIME_TASKS
        forward_tasks = FORWARD_TASKS
        
        # 按分组收集数据
        backward_results = []
        realtime_results = []
        forward_results = []
        
        for pred in self.scored_predictions:
            task = pred.get('task', '')
            if task in backward_tasks:
                backward_results.append(pred)
            elif task in realtime_tasks:
                realtime_results.append(pred)
            elif task in forward_tasks:
                forward_results.append(pred)
        
        report_lines = []
        avg_scores: Dict[str, List[float]] = {
            "backward": [],
            "realtime": [],
            "forward": []
        }
        
        # Backward任务评分
        if len(backward_results) > 0:
            report_lines.append("Evaluate Backward Tracing...")
            task_scores: Dict[str, List[float]] = {}
            for pred in backward_results:
                task = pred.get('task', '')
                if task not in task_scores:
                    task_scores[task] = []
                task_scores[task].append(pred['score'])
            
            for task, scores in task_scores.items():
                accuracy = sum(scores) / len(scores) * 100
                report_lines.append(f"Task: {task}, Acc: {accuracy:.2f}%")
                avg_scores["backward"].append(sum(scores) / len(scores))
            
            if avg_scores["backward"]:
                backward_avg = sum(avg_scores["backward"]) / len(avg_scores["backward"]) * 100
                report_lines.append(f"Backward Avg.: {backward_avg:.2f}%\n")
        
        # Realtime任务评分
        if len(realtime_results) > 0:
            report_lines.append("Evaluate Real-time Visual Perception...")
            task_scores = {}
            for pred in realtime_results:
                task = pred.get('task', '')
                if task not in task_scores:
                    task_scores[task] = []
                task_scores[task].append(pred['score'])
            
            for task, scores in task_scores.items():
                accuracy = sum(scores) / len(scores) * 100
                report_lines.append(f"Task: {task}, Acc: {accuracy:.2f}%")
                avg_scores["realtime"].append(sum(scores) / len(scores))
            
            if avg_scores["realtime"]:
                realtime_avg = sum(avg_scores["realtime"]) / len(avg_scores["realtime"]) * 100
                report_lines.append(f"Realtime Avg.: {realtime_avg:.2f}%\n")
        
        # Forward任务评分
        if len(forward_results) > 0:
            report_lines.append("Evaluate Forward Active Responding...")
            task_scores = {}
            for pred in forward_results:
                task = pred.get('task', '')
                if task not in task_scores:
                    task_scores[task] = []
                task_scores[task].append(pred['score'])
            
            for task, scores in task_scores.items():
                accuracy = sum(scores) / len(scores) * 100
                report_lines.append(f"Task: {task}, Acc: {accuracy:.2f}%")
                avg_scores["forward"].append(sum(scores) / len(scores))
            
            if avg_scores["forward"]:
                forward_avg = sum(avg_scores["forward"]) / len(avg_scores["forward"]) * 100
                report_lines.append(f"Forward Avg.: {forward_avg:.2f}%\n")
        
        # 计算总体准确率（三个分组的平均，对齐官方逻辑）
        group_averages = []
        
        if avg_scores["backward"]:
            backward_avg = sum(avg_scores["backward"]) / len(avg_scores["backward"])
            group_averages.append(backward_avg)
        
        if avg_scores["realtime"]:
            realtime_avg = sum(avg_scores["realtime"]) / len(avg_scores["realtime"])
            group_averages.append(realtime_avg)
            
        if avg_scores["forward"]:
            forward_avg = sum(avg_scores["forward"]) / len(avg_scores["forward"])
            group_averages.append(forward_avg)
        
        # 总体准确率 = 三个分组的平均值（官方逻辑）
        overall_accuracy = sum(group_averages) / len(group_averages) * 100 if group_averages else 0.0
        report_lines.append(f"Overall Accuracy: {overall_accuracy:.2f}%")
        
        report = "\n".join(report_lines)
        return report, overall_accuracy