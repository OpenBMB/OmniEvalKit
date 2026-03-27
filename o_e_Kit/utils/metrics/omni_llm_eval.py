#!/usr/bin/env python3
"""
Omni LLM Evaluation Tool
评估生成内容与GT的相关性以及生成内容的流畅性
"""

import json
import argparse
import os
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import statistics
from .llm_call_new import ChatClient, APIModelName

# 评估提示模板
RELEVANCE_PROMPT_TEMPLATE = """Please evaluate the relevance between the model-generated content and the ground truth annotation. Consider semantic similarity and content matching. Language is not limited (English, Chinese, etc.).

Scoring criteria (1-10 scale):
- 10: Perfect match, semantically identical
- 9: Excellent match, highly consistent content
- 8: Very good match, mostly consistent with minor differences
- 7: Good match, main information consistent
- 6: Decent match, some key information consistent
- 5: Moderate match, partial information consistent
- 4: Fair match, limited information consistent
- 3: Poor match, minimal information consistent
- 2: Very poor match, almost no consistent information
- 1: No match, completely different or irrelevant

Ground Truth: {gt_text}

Model Generated: {pred_text}

Please output only a number between 1-10:"""

FLUENCY_PROMPT_TEMPLATE = """Please evaluate the fluency and naturalness of the following text. Language is not limited (English, Chinese, etc.).

Scoring criteria (1-10 scale):
- 10: Perfect fluency, excellent grammar, crystal clear logic
- 9: Excellent fluency, very good grammar and logic
- 8: Very good fluency, mostly correct grammar with minor issues
- 7: Good fluency, generally correct grammar and logic
- 6: Decent fluency, some grammar or logic issues
- 5: Moderate fluency, noticeable grammar or logic problems
- 4: Fair fluency, several grammar or logic issues
- 3: Poor fluency, many grammar errors or logic problems
- 2: Very poor fluency, serious grammar errors or confused logic
- 1: Extremely poor fluency, severe grammar errors or completely illogical

Text to evaluate: {text}

Please output only a number between 1-10:"""

TIMESTAMP_RELEVANCE_PROMPT_TEMPLATE = """Please evaluate the relevance between model-generated content and ground truth content at different timestamps. Consider event descriptions, actions, and scene matching. Language is not limited (English, Chinese, etc.).

Scoring criteria (1-10 scale):
- 10: Perfect timestamp alignment, events/actions/scenes perfectly match
- 9: Excellent alignment, highly consistent descriptions
- 8: Very good alignment, mostly consistent with minor differences
- 7: Good alignment, main events/actions consistent
- 6: Decent alignment, some key events/actions consistent
- 5: Moderate alignment, partial events/actions consistent
- 4: Fair alignment, limited events/actions consistent
- 3: Poor alignment, minimal events/actions consistent
- 2: Very poor alignment, almost no consistent events/actions
- 1: No alignment, completely different or irrelevant events/actions

Ground Truth Timestamps:
{gt_timestamp_text}

Model Generated Timestamps:
{pred_timestamp_text}

Please output only a number between 1-10:"""


class OmniLLMEvaluator:
    """全模态LLM评估器"""
    
    def __init__(self):
        self.chat_client = ChatClient()
        
    def extract_score(self, response: str) -> int:
        """从LLM响应中提取分数"""
        # 查找1-10之间的数字（包括两位数的10）
        numbers = re.findall(r'\b(?:10|[1-9])\b', response.strip())
        if numbers:
            score = int(numbers[0])
            # 确保分数在1-10范围内
            if 1 <= score <= 10:
                return score
        
        # 如果没找到有效分数，返回默认分数5
        print(f"Warning: Could not extract valid score from response: {response}")
        return 5
    
    def evaluate_relevance(self, gt_text: str, pred_text: str) -> int:
        """评估生成内容与GT的相关性"""
        prompt = RELEVANCE_PROMPT_TEMPLATE.format(
            gt_text=gt_text,
            pred_text=pred_text
        )
        
        try:
            response = self.chat_client.get_eval(content=prompt, max_tokens=10, model_name=APIModelName.GPT_4O_MINI)
            score = self.extract_score(response)
            return score
        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
            return 5  # 十分制的中间值
    
    def evaluate_fluency(self, text: str) -> int:
        """评估文本流畅性"""
        prompt = FLUENCY_PROMPT_TEMPLATE.format(text=text)
        
        try:
            response = self.chat_client.get_eval(content=prompt, max_tokens=10, model_name=APIModelName.GPT_4O_MINI)
            score = self.extract_score(response)
            return score
        except Exception as e:
            print(f"Error in fluency evaluation: {e}")
            return 5  # 十分制的中间值
    
    def evaluate_timestamp_relevance(self, gt_timestamps: List[Dict], pred_timestamps: List[Dict]) -> int:
        """评估时间戳级别的相关性"""
        # 格式化时间戳文本
        gt_formatted = "\n".join([f"时间{item['timestamp']}s: {item['text']}" for item in gt_timestamps])
        pred_formatted = "\n".join([f"时间{item['timestamp']}s: {item['text']}" for item in pred_timestamps])
        
        prompt = TIMESTAMP_RELEVANCE_PROMPT_TEMPLATE.format(
            gt_timestamp_text=gt_formatted,
            pred_timestamp_text=pred_formatted
        )
        
        try:
            response = self.chat_client.get_eval(content=prompt, max_tokens=10, model_name=APIModelName.GPT_4O_MINI)
            score = self.extract_score(response)
            return score
        except Exception as e:
            print(f"Error in timestamp relevance evaluation: {e}")
            return 5  # 十分制的中间值
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """评估单个样本"""
        # 适配新的数据格式
        annotation = sample.get('annotation', {})
        
        # 从annotation中获取GT信息
        gt_text = annotation.get('gt_text', '')
        gt_timestamps = annotation.get('gt_timestamped_text', [])
        
        # 从prediction字段获取预测文本
        pred_text = sample.get('prediction', '')
        
        # 从prediction_full字段获取时间戳信息（如果存在）
        pred_timestamps = []
        if 'prediction_full' in sample:
            pred_full = sample['prediction_full']
            pred_timestamps = pred_full.get('timestamped_generations', [])
        
        results = {
            'sample_id': sample.get('idx', 0),  # 使用idx作为sample_id
        }
        
        # 1. 整体相关性评估
        if gt_text and pred_text:
            relevance_score = self.evaluate_relevance(gt_text, pred_text)
            results['relevance_score'] = relevance_score
        else:
            results['relevance_score'] = 0
        
        # 2. 流畅性评估
        if pred_text:
            fluency_score = self.evaluate_fluency(pred_text)
            results['fluency_score'] = fluency_score
        else:
            results['fluency_score'] = 0
        
        # 3. 时间戳级别相关性评估
        if gt_timestamps and pred_timestamps:
            timestamp_relevance_score = self.evaluate_timestamp_relevance(gt_timestamps, pred_timestamps)
            results['timestamp_relevance_score'] = timestamp_relevance_score
        else:
            results['timestamp_relevance_score'] = 0
        
        return results

    def evaluate_simple_relevance(self, gt_text: str, pred_text: str) -> int:
        """简单的全文相关性评估，专门用于VisionCap_offline2和OmniCap_offline2，十分制评分，支持多语言"""
        simple_prompt = f"""Please evaluate the relevance between the model-generated content and the ground truth annotation. Consider semantic similarity and content matching. Language is not limited (English, Chinese, etc.).

Scoring criteria (1-10 scale):
- 10: Perfect match, semantically identical
- 9: Excellent match, highly consistent content
- 8: Very good match, mostly consistent with minor differences
- 7: Good match, main information consistent
- 6: Decent match, some key information consistent
- 5: Moderate match, partial information consistent
- 4: Fair match, limited information consistent
- 3: Poor match, minimal information consistent
- 2: Very poor match, almost no consistent information
- 1: No match, completely different or irrelevant

Ground Truth: {gt_text}

Model Generated: {pred_text}

Please output only a number between 1-10:"""
        
        try:
            response = self.chat_client.get_eval(content=simple_prompt, max_tokens=10, model_name=APIModelName.GPT_4O_MINI)
            score = self.extract_score(response)
            return score
        except Exception as e:
            print(f"Error in simple relevance evaluation: {e}")
            return 5  # 十分制的中间值
    
    def evaluate_simple_batch(self, predictions: List[Dict]) -> Tuple[List[Dict], str, float]:
        """
        简单批量评估，专门用于VisionCap_offline2和OmniCap_offline2
        只评估全文相关性，不考虑时间戳
        """
        evaluation_results = []
        relevance_scores = []
        
        print(f"开始简单LLM评估，共 {len(predictions)} 个样本...")
        
        for sample in tqdm(predictions, desc="Evaluating samples (simple mode)"):
            try:
                annotation = sample.get('annotation', {})
                gt_text = annotation.get('gt_text', '')
                pred_text = sample.get('prediction', '')
                
                result = {
                    'sample_id': sample.get('idx', 0),
                    'relevance_score': 0
                }
                
                # 只评估整体相关性
                if gt_text and pred_text:
                    relevance_score = self.evaluate_simple_relevance(gt_text, pred_text)
                    result['relevance_score'] = relevance_score
                    relevance_scores.append(relevance_score)
                
                evaluation_results.append(result)
                
            except Exception as e:
                print(f"Error evaluating sample {sample.get('idx', 'unknown')}: {e}")
                continue
        
        # 计算统计结果
        stats = calculate_statistics(relevance_scores)
        
        # 生成简化报告
        report = f"Simple LLM Evaluation Report\n"
        report += f"=" * 60 + "\n"
        report += f"Total evaluated samples: {len(evaluation_results)}\n"
        report += f"Valid relevance scores: {len(relevance_scores)}\n"
        report += f"\n📊 Content Relevance (GT vs Prediction):\n"
        report += f"  Mean: {stats['mean']:.2f}/10\n"  # 改为十分制
        report += f"  Std:  {stats['std']:.2f}\n"
        report += f"  Range: {stats['min']}-{stats['max']}\n"
        
        # 分数分布 - 改为1-10分
        score_dist = {i: relevance_scores.count(i) for i in range(1, 11)}
        report += f"\n📈 Score Distribution:\n"
        for score, count in score_dist.items():
            if count > 0:  # 只显示有样本的分数
                percentage = (count / len(relevance_scores) * 100) if relevance_scores else 0
                report += f"  {score}分: {count} samples ({percentage:.1f}%)\n"
        
        final_score = stats['mean']
        return evaluation_results, report, final_score

    def batch_evaluate(self, predictions: List[Dict]) -> Tuple[List[Dict], str, float]:
        """
        批量评估并生成报告
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            Tuple[评估结果列表, 报告文本, 最终分数]
        """
        evaluation_results = []
        relevance_scores = []
        fluency_scores = []
        timestamp_relevance_scores = []
        
        print(f"开始LLM评估，共 {len(predictions)} 个样本...")
        
        for sample in tqdm(predictions, desc="Evaluating samples"):
            try:
                result = self.evaluate_sample(sample)
                evaluation_results.append(result)
                
                if result['relevance_score'] > 0:
                    relevance_scores.append(result['relevance_score'])
                if result['fluency_score'] > 0:
                    fluency_scores.append(result['fluency_score'])
                if result['timestamp_relevance_score'] > 0:
                    timestamp_relevance_scores.append(result['timestamp_relevance_score'])
                    
            except Exception as e:
                print(f"Error evaluating sample {sample.get('idx', 'unknown')}: {e}")
                continue
        
        # 计算统计结果
        stats = {
            'relevance': calculate_statistics(relevance_scores),
            'fluency': calculate_statistics(fluency_scores),
            'timestamp_relevance': calculate_statistics(timestamp_relevance_scores)
        }
        
        # 生成报告
        report = f"LLM Evaluation Report\n"
        report += f"=" * 60 + "\n"
        report += f"Total evaluated samples: {len(evaluation_results)}\n"
        report += f"\n📊 Overall Relevance (GT vs Prediction):\n"
        report += f"  Mean: {stats['relevance']['mean']}/10\n"  # 改为十分制
        report += f"  Std:  {stats['relevance']['std']}\n"
        report += f"  Range: {stats['relevance']['min']}-{stats['relevance']['max']}\n"
        
        report += f"\n✍️  Fluency (Prediction Quality):\n"
        report += f"  Mean: {stats['fluency']['mean']}/10\n"   # 改为十分制
        report += f"  Std:  {stats['fluency']['std']}\n"
        report += f"  Range: {stats['fluency']['min']}-{stats['fluency']['max']}\n"
        
        report += f"\n⏰ Timestamp Relevance (Time-aligned Content):\n"
        report += f"  Mean: {stats['timestamp_relevance']['mean']}/10\n"  # 改为十分制
        report += f"  Std:  {stats['timestamp_relevance']['std']}\n"
        report += f"  Range: {stats['timestamp_relevance']['min']}-{stats['timestamp_relevance']['max']}\n"
        
        # 使用相关性平均分作为最终分数
        final_score = stats['relevance']['mean']
        
        return evaluation_results, report, final_score


def load_predictions(input_path: str) -> List[Dict]:
    """加载预测结果，支持JSON和JSONL格式"""
    samples = []
    
    # 检查文件扩展名，决定加载方式
    if input_path.endswith('.jsonl'):
        # JSONL格式（原有格式）
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        # JSON格式（新格式）
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            else:
                # 如果是单个对象，转换为列表
                samples = [data]
    
    return samples


def calculate_statistics(scores: List[int]) -> Dict:
    """计算统计信息"""
    if not scores:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': round(statistics.mean(scores), 2),
        'std': round(statistics.stdev(scores) if len(scores) > 1 else 0, 2),
        'min': min(scores),
        'max': max(scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Omni LLM Evaluation Tool")
    parser.add_argument("--input_file", type=str, default="./results/VisionCap.json",
                        help="输入的JSON/JSONL预测文件路径")
    parser.add_argument("--output_dir", type=str, default="./results/omni_eval",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数量（用于测试）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载预测数据
    print(f"Loading predictions from: {args.input_file}")
    samples = load_predictions(args.input_file)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for evaluation")
    
    print(f"Total samples to evaluate: {len(samples)}")
    
    # 初始化评估器
    evaluator = OmniLLMEvaluator()
    
    # 评估所有样本
    print("Starting LLM evaluation...")
    evaluation_results = []
    relevance_scores = []
    fluency_scores = []
    timestamp_relevance_scores = []
    
    for sample in tqdm(samples, desc="Evaluating samples"):
        try:
            result = evaluator.evaluate_sample(sample)
            evaluation_results.append(result)
            
            if result['relevance_score'] > 0:
                relevance_scores.append(result['relevance_score'])
            if result['fluency_score'] > 0:
                fluency_scores.append(result['fluency_score'])
            if result['timestamp_relevance_score'] > 0:
                timestamp_relevance_scores.append(result['timestamp_relevance_score'])
                
        except Exception as e:
            print(f"Error evaluating sample {sample.get('sample_id', 'unknown')}: {e}")
            continue
    
    # 计算统计结果
    print("\nCalculating statistics...")
    stats = {
        'relevance': calculate_statistics(relevance_scores),
        'fluency': calculate_statistics(fluency_scores),
        'timestamp_relevance': calculate_statistics(timestamp_relevance_scores)
    }
    
    # 保存详细结果
    results_file = os.path.join(args.output_dir, "detailed_results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in evaluation_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 保存统计结果
    stats_file = os.path.join(args.output_dir, "statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n{'='*60}")
    print("LLM Evaluation Results")
    print(f"{'='*60}")
    print(f"Total evaluated samples: {len(evaluation_results)}")
    print(f"\n📊 Overall Relevance (GT vs Prediction):")
    print(f"  Mean: {stats['relevance']['mean']}/10")  # 改为十分制
    print(f"  Std:  {stats['relevance']['std']}")
    print(f"  Range: {stats['relevance']['min']}-{stats['relevance']['max']}")
    
    print(f"\n✍️  Fluency (Prediction Quality):")
    print(f"  Mean: {stats['fluency']['mean']}/10")   # 改为十分制
    print(f"  Std:  {stats['fluency']['std']}")
    print(f"  Range: {stats['fluency']['min']}-{stats['fluency']['max']}")
    
    print(f"\n⏰ Timestamp Relevance (Time-aligned Content):")
    print(f"  Mean: {stats['timestamp_relevance']['mean']}/10")  # 改为十分制
    print(f"  Std:  {stats['timestamp_relevance']['std']}")
    print(f"  Range: {stats['timestamp_relevance']['min']}-{stats['timestamp_relevance']['max']}")
    
    print(f"\n📁 Results saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Statistics: {stats_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 