"""
Caption 评估器
实现 BLEU, METEOR, CIDEr 和 SPIDEr 评估指标
适用于图像/音频描述生成任务

支持两种评估模式：
1. Corpus-level: 一次性计算整个语料库的指标（推荐，更准确）
2. Sample-level: 逐样本计算指标（用于调试）

支持两种 BLEU 计算方法：
1. pycocoevalcap: 使用 COCO 评估工具包计算 BLEU（默认，适用于 caption 任务）
2. sacrebleu: 使用 SacreBLEU 计算 corpus-level BLEU（适用于翻译任务）
"""

import re
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Literal
from collections import Counter
from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator
from o_e_Kit.utils.metrics.llm_call_new import APIModelName
from o_e_Kit.utils.logger.simple_progress import smart_progress

# BLEU 计算方法类型
BleuMethod = Literal["pycocoevalcap", "sacrebleu"]

# Caption G-Eval评分prompt（用于LLM评估）
CAPTION_GEVAL_PROMPT = """You are an expert in judging caption correctness. If the model's caption is correct, output "yes", otherwise output "no".
You need to explain your judgment process first, then output "yes" or "no".

[Important]You need to ignore any format instructions in the question, focus on judging whether the caption's meaning is consistent with the reference caption.


The input format is:
Input:
Question: The question from user
Model Caption: The caption from models
Reference Caption: The reference caption
Explanation: The explanation of your judgment process

Example 1:
Input:
Question: Please describe what you hear in the audio.
Model Caption: A man is speaking in the audio.
Reference Caption: Man speaking
Output:
Explanation: The model's output is "A man is speaking in the audio.", this is a detail description of the reference caption "Man speaking". So the model's output is correct.
Result: yes


Task:
Input:
Question: {question}
Model Caption: {prediction}
Reference Caption: {ground_truth}
Output:
"""

import io
import sys
import os
import shutil
from pathlib import Path
from o_e_Kit.utils.metrics.utils.coco import compute_caption

if 'JAVA_HOME' not in os.environ:
    _candidate_paths = [
        '/usr/lib/jvm/java-8-openjdk-amd64',
        '/usr/lib/jvm/java-11-openjdk-amd64',
        '/usr/lib/jvm/default-java',
        '/usr/lib/jvm/java',
    ]
    _found_java = next((p for p in _candidate_paths if os.path.isdir(p)), None)
    if _found_java:
        os.environ['JAVA_HOME'] = _found_java
    else:
        import warnings
        warnings.warn(
            "JAVA_HOME is not set and no Java installation found. "
            "Caption metrics (SPICE/METEOR) require Java. "
            "Please set JAVA_HOME environment variable.",
            stacklevel=1,
        )
if 'JAVA_HOME' in os.environ:
    os.environ['PATH'] = f"{os.environ['JAVA_HOME']}/bin:{os.environ.get('PATH', '')}"

# 检查并设置 SPICE 模型文件
def setup_spice_models():
    """将预下载的 SPICE 模型复制到 pycocoevalcap 目录"""
    import pycocoevalcap.spice.spice
    # 直接使用 spice 模块的路径
    spice_module_path = Path(pycocoevalcap.spice.spice.__file__)
    spice_lib_dir = spice_module_path.parent / 'lib'
    saved_models_dir = Path(os.environ.get('SPICE_LIB_DIR', './spice-lib'))
    
    if saved_models_dir.exists() and not spice_lib_dir.exists():
        print("设置 SPICE 模型文件...")
        spice_lib_dir.mkdir(parents=True, exist_ok=True)
        for model_file in saved_models_dir.glob('*.jar'):
            target_file = spice_lib_dir / model_file.name
            if not target_file.exists():
                shutil.copy2(model_file, target_file)
        print("SPICE 模型文件设置完成")
    elif saved_models_dir.exists() and spice_lib_dir.exists():
        # 检查是否所有文件都存在
        model_files = list(saved_models_dir.glob('*.jar'))
        missing_files = []
        for model_file in model_files:
            target_file = spice_lib_dir / model_file.name
            if not target_file.exists():
                missing_files.append(model_file)
        
        if missing_files:
            print(f"补充缺失的 SPICE 模型文件 ({len(missing_files)} 个)...")
            for model_file in missing_files:
                target_file = spice_lib_dir / model_file.name
                shutil.copy2(model_file, target_file)
            print("SPICE 模型文件补充完成")

# 初始化时设置模型

class CaptionEvaluator(BaseEvaluator):
    """
    Caption 评估器
    用于评估生成的描述文本质量
    
    评估指标：
    - BLEU (1-4): 基于n-gram的精确度（Corpus-level）
    - METEOR: 考虑同义词的召回率和精确度
    - ROUGE-L: 基于最长公共子序列的召回率
    - CIDEr: 基于TF-IDF的共识度量（Corpus-level）
    - SPICE: 基于语义图的相似度
    - SPIDEr: (CIDEr + SPICE) / 2
    
    评估模式：
    - evaluate(): 批量评估，计算真正的 Corpus-level 指标（推荐）
    - eval(): 单样本评估，用于 LLM fallback
    
    BLEU 计算方法：
    - pycocoevalcap: 使用 COCO 评估工具包（默认，适用于 caption 任务）
    - sacrebleu: 使用 SacreBLEU（适用于翻译任务，如 CoVoST2）
    """
    
    def __init__(self, use_llm_fallback: bool = True, bleu_method: BleuMethod = "sacrebleu", target_lang: str = "en"):
        """
        Args:
            use_llm_fallback: 是否使用LLM作为后备
            bleu_method: BLEU 计算方法，可选 "pycocoevalcap" 或 "sacrebleu"
            target_lang: 目标语言，用于 sacrebleu 选择 tokenizer（"zh" 用中文分词，其他用 "13a"）
        """
        super().__init__(use_llm_fallback)
        self.bleu_method = bleu_method
        self.target_lang = target_lang
        self.scores = {
            'Bleu_1': [],
            'Bleu_2': [],
            'Bleu_3': [],
            'Bleu_4': [],
            'METEOR': [],
            'CIDEr': [],
            'ROUGE_L': [],
            'SPICE': [],
            'SPIDEr': []
        }
        # Corpus-level 分数（由 evaluate() 计算）
        self.corpus_scores = {}
        
        # 只在使用 pycocoevalcap 时设置 SPICE 模型
        if bleu_method == "pycocoevalcap":
            setup_spice_models()

    def reset(self):
        super().reset()
        self.scores = {
            'Bleu_1': [],
            'Bleu_2': [],
            'Bleu_3': [],
            'Bleu_4': [],
            'METEOR': [],
            'CIDEr': [],
            'ROUGE_L': [],
            'SPICE': [],
            'SPIDEr': []
        }
        self.corpus_scores = {}
    
    def _compute_sacrebleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        使用 SacreBLEU 计算 corpus-level BLEU 分数
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表（每个样本可有多个参考）
        
        Returns:
            包含 BLEU 分数的字典
        """
        try:
            import sacrebleu
        except ImportError:
            print("❌ sacrebleu not installed. Please run: pip install sacrebleu")
            return {}
        
        # SacreBLEU 期望 references 的格式是：[[ref1_for_sample1, ref1_for_sample2, ...], [ref2_for_sample1, ...]]
        # 我们的格式是：[[refs_for_sample1], [refs_for_sample2], ...]
        # 需要转置参考列表
        max_refs = max(len(refs) for refs in references)
        transposed_refs = []
        for ref_idx in range(max_refs):
            ref_list = []
            for sample_refs in references:
                if ref_idx < len(sample_refs):
                    ref_list.append(sample_refs[ref_idx])
                else:
                    # 如果该样本没有足够的参考，使用第一个参考
                    ref_list.append(sample_refs[0])
            transposed_refs.append(ref_list)
        
        # 根据目标语言选择 tokenizer
        # zh: 中文分词，适用于中文目标
        # 13a: 默认的 Moses tokenizer，适用于英语等语言
        if self.target_lang == "zh":
            tokenizer = "zh"
        else:
            tokenizer = "13a"
        
        # 计算 BLEU
        bleu = sacrebleu.corpus_bleu(predictions, transposed_refs, tokenize=tokenizer)
        
        metrics = {
            'Bleu_1': bleu.precisions[0] / 100.0,  # sacrebleu 返回百分比
            'Bleu_2': bleu.precisions[1] / 100.0,
            'Bleu_3': bleu.precisions[2] / 100.0,
            'Bleu_4': bleu.precisions[3] / 100.0,
            'BLEU': bleu.score / 100.0,  # 总体 BLEU 分数
            'tokenizer': tokenizer,  # 记录使用的 tokenizer
        }
        
        # 尝试计算 chrF（字符级 F-score，对中文更友好）
        try:
            chrf = sacrebleu.corpus_chrf(predictions, transposed_refs)
            metrics['chrF'] = chrf.score / 100.0
        except Exception:
            pass
        
        return metrics
    
    def _extract_pred_and_ref(self, prediction: Dict[str, Any]) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        从 prediction 字典中提取预测文本和参考答案
        
        Returns:
            (pred_text, references) 或 (None, None) 如果提取失败
        """
        pred_text = str(prediction.get('prediction', ''))
        if not pred_text.strip():
            return None, None
        
        # 支持多种数据格式提取参考答案
        references = None
        if 'ground_truth' in prediction:
            references = prediction['ground_truth']
        elif 'annotation' in prediction and prediction['annotation']:
            references = prediction['annotation'].get('caption', 
                            prediction['annotation'].get('reference', 
                                prediction['annotation'].get('gt_answer', '')))
        elif 'reference' in prediction:
            references = prediction['reference']
        elif 'caption' in prediction:
            references = prediction['caption']
        
        if references:
            references = references.replace("Based on the audio you provided, the main content of this audio has the following possibilities:\n", "")
        
        if not references:
            return None, None
        
        # 将 references 转换为列表
        if not isinstance(references, list):
            references = [str(references)]
        else:
            references = [str(ref) for ref in references]
        
        return pred_text, references
    
    def evaluate(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量评估：计算真正的 Corpus-level BLEU/CIDEr/SPICE 等指标
        
        这是推荐的评估方式，因为：
        1. BLEU 的 n-gram 统计在全局计算才有意义
        2. CIDEr 的 TF-IDF 权重需要全局语料计算
        3. SPICE 模型只需加载一次，大幅提升性能
        
        根据 bleu_method 选择不同的计算方式：
        - pycocoevalcap: 使用 COCO 工具包（计算 BLEU/METEOR/CIDEr/SPICE/SPIDEr）
        - sacrebleu: 使用 SacreBLEU（计算 BLEU/chrF，适用于翻译任务）
        
        Args:
            predictions: 预测列表
        
        Returns:
            评分后的预测列表（每个样本获得相同的 corpus-level 分数）
        """
        self.reset()
        
        # 1. 收集所有有效的 predictions 和 references
        all_preds = []
        all_refs = []
        valid_indices = []
        failed_indices = []
        
        print("📊 Collecting predictions and references...")
        for i, pred in enumerate(smart_progress(predictions, desc="Extracting")):
            pred_text, references = self._extract_pred_and_ref(pred)
            if pred_text is not None and references is not None:
                all_preds.append(pred_text)
                all_refs.append(references)
                valid_indices.append(i)
            else:
                failed_indices.append(i)
                predictions[i]['score'] = 0.0
                predictions[i]['eval_method'] = 'failed'
                predictions[i]['eval_failed'] = True
                self.failed_count += 1
                self.total_samples += 1
        
        print(f"✅ Collected {len(all_preds)} valid samples, {len(failed_indices)} failed")
        
        if not all_preds:
            self.scored_predictions = predictions
            return predictions
        
        # 2. 根据 bleu_method 选择计算方式
        if self.bleu_method == "sacrebleu":
            # 使用 SacreBLEU 计算（适用于翻译任务）
            print("🔄 Computing Corpus-level metrics using SacreBLEU...")
            metrics = self._compute_sacrebleu(all_preds, all_refs)
            
            if not metrics:
                print("❌ SacreBLEU computation failed!")
                for i in valid_indices:
                    predictions[i]['score'] = 0.0
                    predictions[i]['eval_method'] = 'failed'
                    predictions[i]['eval_failed'] = True
                    self.failed_count += 1
                    self.total_samples += 1
                self.scored_predictions = predictions
                return predictions
        else:
            # 使用 pycocoevalcap 计算（默认，适用于 caption 任务）
            print("🔄 Computing Corpus-level metrics using pycocoevalcap (BLEU, METEOR, CIDEr, SPICE)...")
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                metrics = compute_caption(all_refs, all_preds)
            finally:
                sys.stdout = old_stdout
            
            if not metrics:
                print("❌ Corpus-level computation failed, falling back to sample-level...")
                # 回退到逐样本评估
                for i in valid_indices:
                    result = self.eval(predictions[i])
                    if result is not None:
                        self.rule_eval_count += 1
                        result['eval_method'] = 'rule'
                    else:
                        predictions[i]['score'] = 0.0
                        predictions[i]['eval_method'] = 'failed'
                        predictions[i]['eval_failed'] = True
                        self.failed_count += 1
                    self.total_samples += 1
                self.scored_predictions = predictions
                return predictions
        
        # 3. 存储 Corpus-level 分数
        self.corpus_scores = metrics
        print(f"✅ Corpus-level metrics computed (method: {self.bleu_method}):")
        for metric, score in metrics.items():
            if isinstance(score, (int, float)):
                print(f"   {metric}: {score:.4f}")
            else:
                print(f"   {metric}: {score}")
        
        # 4. 为每个有效样本设置 corpus 分数
        # 根据不同方法选择主分数
        if self.bleu_method == "sacrebleu":
            main_score = metrics.get('BLEU', metrics.get('Bleu_4', 0.0))
        else:
            main_score = metrics.get('SPIDEr', metrics.get('CIDEr', 0.0))
        
        for i in valid_indices:
            predictions[i]['scores'] = metrics.copy()
            predictions[i]['corpus_scores'] = metrics.copy()  # 明确标记为 corpus-level
            predictions[i]['score'] = main_score
            predictions[i]['eval_method'] = 'rule'
            predictions[i]['bleu_method'] = self.bleu_method
            self.rule_eval_count += 1
            self.total_samples += 1
        
        # 5. 更新 scores 列表（用于 summary）
        for metric_name, score in metrics.items():
            if metric_name in self.scores:
                self.scores[metric_name].append(score)
        
        self.scored_predictions = predictions
        print(f"✅ Evaluation complete! Total: {self.total_samples}, Valid: {len(valid_indices)}, Failed: {len(failed_indices)}")
        
        return predictions
    
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        单样本规则评估：计算 BLEU, METEOR, CIDEr, SPIDEr 等指标
        
        注意：此方法计算的是 sample-level 指标，用于 LLM fallback 或调试。
        对于完整评估，推荐使用 evaluate() 方法获取真正的 Corpus-level 指标。
        """
        pred_text, references = self._extract_pred_and_ref(prediction)
        
        if pred_text is None or references is None:
            return None
        
        # 暂时捕获标准输出，因为 compute_caption 会打印很多信息
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            metrics = compute_caption([references], [pred_text])
        finally:
            sys.stdout = old_stdout
        
        if not metrics:
            return None
        
        # 保存分数到累计列表（sample-level）
        for metric_name, score in metrics.items():
            if metric_name in self.scores:
                self.scores[metric_name].append(score)
        
        # 返回结果
        prediction['scores'] = metrics
        prediction['sample_level'] = True  # 标记为 sample-level
        
        if 'SPIDEr' in metrics:
            prediction['score'] = metrics['SPIDEr']
        elif 'CIDEr' in metrics:
            prediction['score'] = metrics['CIDEr']
        else:
            prediction['score'] = 0.0
        
        return prediction
            
    
    def evaluate_single(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        覆盖基类方法，对于Caption任务，总是使用规则评估
        """
        self.total_samples += 1
        
        # 首先尝试规则评估（计算各种指标）
        result = self.eval(prediction)
        
        if result is not None:
            self.rule_eval_count += 1
            result['eval_method'] = 'rule'
            return result
        
        # 如果规则评估失败且启用了LLM后备
        if self.use_llm_fallback and self.llm_client:
            result = self.llm_eval(prediction)
            if result is not None:
                self.llm_eval_count += 1
                result['eval_method'] = 'llm'
                return result
        
        # 如果都失败了
        self.failed_count += 1
        prediction['eval_method'] = 'failed'
        prediction['score'] = 0.0
        prediction['eval_failed'] = True
        return prediction
    
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用LLM评估答案正确性
        与RefQA保持一致
        """
        if not self.llm_client:
            return None
        
        pred_text = str(prediction.get('prediction', ''))
        
        # 支持多种数据格式提取参考答案
        references = None
        if 'ground_truth' in prediction:
            references = prediction['ground_truth']
        elif 'annotation' in prediction and prediction['annotation']:
            references = prediction['annotation'].get('caption', 
                            prediction['annotation'].get('reference', 
                                prediction['annotation'].get('gt_answer', '')))
        elif 'reference' in prediction:
            references = prediction['reference']
        elif 'caption' in prediction:
            references = prediction['caption']
        
        if not references:
            return None
        
        # 将references转换为列表
        if not isinstance(references, list):
            references_list = [str(references)]
        else:
            references_list = [str(ref) for ref in references]
        
        # 支持多种数据格式提取question
        question = ''
        if 'question' in prediction:
            question = prediction['question']
        elif 'annotation' in prediction and prediction['annotation']:
            question = prediction['annotation'].get('prompt', prediction['annotation'].get('question', ''))
        
        # 逐个评估每个可能的参考答案
        is_correct = False
        matched_reference = None
        llm_responses = []
        
        for ref in references_list:
            # 使用预定义的G-Eval prompt模板
            prompt = CAPTION_GEVAL_PROMPT.format(
                question=question if question else "Generate a caption for the given content.",
                prediction=pred_text,
                ground_truth=ref
            )
            
            try:
                response = self.llm_client.get_eval(content=prompt, max_tokens=1024, model_name=APIModelName.GPT_4O_MINI, temperature=0)
                response_lower = response.lower()
                llm_responses.append(f"Ref: {ref} - Response: {response.strip()}")
                
                # 提取yes/no判断
                score = response_lower.strip().split("\n")[-1]
                if "yes" in score:
                    is_correct = True
                    matched_reference = ref
                    break
                elif "no" not in score:
                    print(f"Warning: LLM response does not contain yes/no for reference '{ref}': {response}")
                    
            except Exception as e:
                print(f"Error calling LLM for reference '{ref}': {e}")
                continue
        
        # 如果LLM认为正确，给较高的分数
        if is_correct:
            prediction['score'] = 1  # LLM评估给0.8分
        else:
            prediction['score'] = 0  # LLM评估认为不正确给0.2分
        
        prediction['llm_match'] = is_correct
        if matched_reference:
            prediction['matched_reference'] = matched_reference
        prediction['llm_responses'] = llm_responses
        
        # 即使LLM评估，也计算传统指标
        self.eval(prediction)
        
        return prediction
    
    def summary(self) -> Tuple[str, float]:
        """
        生成评估摘要
        
        优先使用 Corpus-level 分数（由 evaluate() 计算）
        如果没有 corpus 分数，则使用 sample-level 平均分数
        """
        stats = self.get_eval_stats()
        
        report = f"Caption Evaluation Report\n"
        report += f"{'=' * 50}\n"
        report += f"Total Samples: {self.total_samples}\n"
        report += f"BLEU Method: {self.bleu_method}\n"
        
        # 优先使用 Corpus-level 分数
        if self.corpus_scores:
            if self.bleu_method == "sacrebleu":
                # SacreBLEU 模式的输出
                report += f"\n📊 Corpus-level Metric Scores (SacreBLEU):\n"
                report += f"  BLEU (corpus): {self.corpus_scores.get('BLEU', 0.0):.4f}\n"
                report += f"  BLEU-1: {self.corpus_scores.get('Bleu_1', 0.0):.4f}\n"
                report += f"  BLEU-2: {self.corpus_scores.get('Bleu_2', 0.0):.4f}\n"
                report += f"  BLEU-3: {self.corpus_scores.get('Bleu_3', 0.0):.4f}\n"
                report += f"  BLEU-4: {self.corpus_scores.get('Bleu_4', 0.0):.4f}\n"
                if 'chrF' in self.corpus_scores:
                    report += f"  chrF:   {self.corpus_scores.get('chrF', 0.0):.4f}\n"
                
                final_score = self.corpus_scores.get('BLEU', self.corpus_scores.get('Bleu_4', 0.0))
            else:
                # pycocoevalcap 模式的输出
                report += f"\n📊 Corpus-level Metric Scores (pycocoevalcap):\n"
                report += f"  BLEU-1: {self.corpus_scores.get('Bleu_1', 0.0):.4f}\n"
                report += f"  BLEU-2: {self.corpus_scores.get('Bleu_2', 0.0):.4f}\n"
                report += f"  BLEU-3: {self.corpus_scores.get('Bleu_3', 0.0):.4f}\n"
                report += f"  BLEU-4: {self.corpus_scores.get('Bleu_4', 0.0):.4f}\n"
                report += f"  METEOR: {self.corpus_scores.get('METEOR', 0.0):.4f}\n"
                report += f"  ROUGE-L: {self.corpus_scores.get('ROUGE_L', 0.0):.4f}\n"
                report += f"  CIDEr:  {self.corpus_scores.get('CIDEr', 0.0):.4f}\n"
                if 'SPICE' in self.corpus_scores and self.corpus_scores['SPICE'] > 0:
                    report += f"  SPICE:  {self.corpus_scores.get('SPICE', 0.0):.4f}\n"
                report += f"  SPIDEr: {self.corpus_scores.get('SPIDEr', 0.0):.4f}\n"
                
                final_score = self.corpus_scores.get('SPIDEr', self.corpus_scores.get('CIDEr', 0.0))
        else:
            # 回退到 sample-level 平均分数
            avg_scores = {}
            for metric, scores in self.scores.items():
                if scores:
                    avg_scores[metric] = np.mean(scores)
                else:
                    avg_scores[metric] = 0.0
            
            report += f"\n📊 Sample-level Average Scores:\n"
            report += f"  (Note: For accurate BLEU/CIDEr, use evaluate() for corpus-level)\n"
            report += f"  BLEU-1: {avg_scores.get('Bleu_1', 0.0):.4f}\n"
            report += f"  BLEU-2: {avg_scores.get('Bleu_2', 0.0):.4f}\n"
            report += f"  BLEU-3: {avg_scores.get('Bleu_3', 0.0):.4f}\n"
            report += f"  BLEU-4: {avg_scores.get('Bleu_4', 0.0):.4f}\n"
            report += f"  METEOR: {avg_scores.get('METEOR', 0.0):.4f}\n"
            report += f"  ROUGE-L: {avg_scores.get('ROUGE_L', 0.0):.4f}\n"
            report += f"  CIDEr:  {avg_scores.get('CIDEr', 0.0):.4f}\n"
            if 'SPICE' in avg_scores and avg_scores['SPICE'] > 0:
                report += f"  SPICE:  {avg_scores.get('SPICE', 0.0):.4f}\n"
            report += f"  SPIDEr: {avg_scores.get('SPIDEr', 0.0):.4f}\n"
            
            final_score = avg_scores.get('SPIDEr', avg_scores.get('CIDEr', 0.0))
        
        report += f"\nEvaluation Method Stats:\n"
        report += f"  Rule-based: {stats['rule_eval_count']} ({stats['rule_eval_rate']:.1%})\n"
        report += f"  LLM-based: {stats['llm_eval_count']} ({stats['llm_eval_rate']:.1%})\n"
        report += f"  Failed: {stats['failed_count']} ({stats['failed_rate']:.1%})\n"
        
        return report, final_score
