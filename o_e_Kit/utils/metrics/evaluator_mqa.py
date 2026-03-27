"""
MQA (Multiple Choice QA) 评估器
参考了 UltraEval-Audio 和 VLMEvalKit 的实现
支持规则匹配、NVEmbed语义匹配、LLM后备三种方法
"""

# MQA选项提取prompt（用于LLM后备方法）
MQA_EXTRACTION_PROMPT = """You are an AI assistant who will help me to match an answer with several options of a single-choice question.
You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer.
If the meaning of all options are significantly different from the answer, output Z.
Your should output a single uppercase character in {choices} (if they are valid options), and Z.

Example 1:
Question: What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
Answer: a cute teddy bear
Your output: A

Example 2:
Question: What is the main object in image?
Options: A. teddy bear B. rabbit C. cat D. dog
Answer: Spider
Your output: Z

Example 3:
Question: {question}
Options:
{options}
Answer: {prediction}
Your output: """

import os
import re
import string
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from o_e_Kit.utils.metrics.evaluator_base import BaseEvaluator
from o_e_Kit.utils.metrics.llm_call_new import APIModelName


def normalize_for_matching(s: str) -> str:
    """
    标准化字符串用于匹配，处理以下差异：
    - 大小写差异
    - 连字符/下划线与空格的差异（如 "middle-aged" vs "middle aged"）
    - 多余空格
    """
    s = str(s).upper()
    s = s.replace('-', ' ').replace('_', ' ')  # 移除连字符和下划线
    return ' '.join(s.split())  # 统一空格


def load_sentence_transformer(device='cuda'):
    """
    加载 Sentence Transformer 模型用于语义相似度匹配
    
    推荐模型：
    - 'Qwen/Qwen3-Embedding-0.6B' - 中英文混合，轻量高效（默认）
    - 'all-MiniLM-L6-v2' - 纯英文，快速
    - 'BAAI/bge-m3' - 多语言支持
    """
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        print(f"Loading Sentence Transformer model for MQA evaluation: {model_name}...")
        model = SentenceTransformer(model_name, device=device)
        model = model.to(torch.bfloat16)
        print(f"Sentence Transformer model loaded successfully for MQA on {device}!")
        return model
    except ImportError:
        print("Error: sentence-transformers not installed. Please run:")
        print("  pip install sentence-transformers==2.7.0")
        return None
    except Exception as e:
        print(f"Failed to load Sentence Transformer model: {e}")
        print("Falling back to rule-based matching for MQA")
        return None


class MQAEvaluator(BaseEvaluator):
    """
    Multiple Choice QA (多选题) 评估器
    支持A/B/C/D等选项的多选题评估
    结合了规则提取、Sentence Transformer语义匹配和LLM提取三种方法
    
    Args:
        use_llm_fallback: 是否使用LLM作为后备方法
        ignore_case: 是否忽略大小写
        use_sentence_transformer: 是否使用Sentence Transformer语义匹配（推荐，在规则匹配失败后使用）
        device: Sentence Transformer运行设备
    """
    
    def __init__(self, use_llm_fallback: bool = True, ignore_case: bool = True, 
                 use_sentence_transformer: bool = False, device: str = 'cuda',
                 **kwargs):
        super().__init__(use_llm_fallback, **kwargs)
        self.ignore_case = ignore_case
        self.correct_samples = 0
        self.extraction_templates = self._build_extraction_templates()
        
        # Sentence Transformer 配置
        self.use_sentence_transformer = use_sentence_transformer
        self.device = device
        self.sentence_transformer = None
        self.st_success_count = 0
        self.st_fallback_count = 0
        
        # 如果启用 Sentence Transformer，加载模型
        if use_sentence_transformer:
            self.sentence_transformer = load_sentence_transformer(device)
            if self.sentence_transformer is None:
                print("Warning: Sentence Transformer model failed to load for MQA, will skip semantic matching")
                self.use_sentence_transformer = False
    
    def reset(self):
        super().reset()
        self.correct_samples = 0
        self.st_success_count = 0
        self.st_fallback_count = 0
    
    def _build_extraction_templates(self) -> List[str]:
        """构建答案提取模板（参考UltraEval-Audio）"""
        templates = [
            # 中文模板
            "答案是[CHOICE]",
            "答案是:[CHOICE]",
            "答案是：[CHOICE]",
            "答案是 [CHOICE]",
            "答案是选项[CHOICE]",
            "答案应该是[CHOICE]",
            "答案选[CHOICE]",
            "[CHOICE]是正确",
            "选项[CHOICE]是最合适的",
            "正确答案是[CHOICE]",
            "应该选[CHOICE]",
            "选择[CHOICE]",
            
            # 英文模板
            "the answer is [CHOICE]",
            "the correct answer is [CHOICE]",
            "answer is [CHOICE]",
            "answer: [CHOICE]",
            "the answer is ([CHOICE])",
            "the answer is option [CHOICE]",
            "would select [CHOICE]",
            "would choose [CHOICE]",
            "would select option [CHOICE]",
            "would choose option [CHOICE]",
            "[CHOICE] is the best answer",
            "[CHOICE] is correct",
            "option [CHOICE] is correct",
            "the best answer is [CHOICE]",
            "the best option is [CHOICE]",
            'the answer is "[CHOICE]"',
            "the answer is '[CHOICE]'",
            "answer is: [CHOICE]",
            "answer is ([CHOICE])",
            "([CHOICE])",
            "option [CHOICE]",
            
            # 特殊格式
            "**[CHOICE]**",
            "**[CHOICE].",
            "**[CHOICE])",
            "is [CHOICE]",
            "is: [CHOICE]",
            "would be [CHOICE]",
            "is option [CHOICE]",
            ": [CHOICE]",
            "\\boxed{[CHOICE]}",
            "\\text{[CHOICE]}",
        ]
        
        # 添加一些变体
        expanded_templates = []
        for template in templates:
            expanded_templates.append(template)
            # 添加带标点的版本
            expanded_templates.append(template + ".")
            expanded_templates.append(template + ",")
            expanded_templates.append(template + ":")
            expanded_templates.append(template + ")")
        
        return expanded_templates
    
    def _normalize_text(self, text: str) -> str:
        """规范化文本"""
        if self.ignore_case:
            text = text.lower()
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def _extract_by_templates(self, text: str, choices: Dict[str, str]) -> Optional[str]:
        """使用模板提取答案"""
        text = self._normalize_text(text)
        
        # 尝试所有模板
        for template in self.extraction_templates:
            for choice in choices.keys():
                pattern = template.replace("[CHOICE]", choice.lower() if self.ignore_case else choice)
                if pattern in text:
                    return choice.upper()
        
        return None
    
    def _extract_by_patterns(self, text: str, choices: Dict[str, str]) -> Optional[str]:
        """使用正则表达式提取答案"""
        text = self._normalize_text(text)
        
        # 检查简单的单字母答案
        if len(text) <= 3:
            text_upper = text.upper()
            for choice in choices.keys():
                if choice in text_upper:
                    return choice
        
        # 检查开头是否是答案
        for choice in choices.keys():
            choice_lower = choice.lower() if self.ignore_case else choice
            # 检查各种开头模式
            if text.startswith(choice_lower):
                # 后面跟着标点或空格
                if len(text) == 1 or (len(text) > 1 and text[1] in '.,;:) \n'):
                    return choice.upper()
        
        # 检查括号中的答案
        patterns = [
            r'\(([A-D])\)',  # (A)
            r'\[([A-D])\]',  # [A]
            r'（([A-D])）',  # 中文括号
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                for match in matches:
                    if match in choices:
                        return match
        
        return None
    
    def _extract_by_content_matching(self, text: str, choices: Dict[str, str]) -> Optional[str]:
        """通过内容匹配提取答案（参考VLMEvalKit）"""
        text = self._normalize_text(text)
        
        # 检查是否包含选项内容
        for choice, content in choices.items():
            if not content:
                continue
            
            # 确保 content 是字符串类型，避免 float/int 调用字符串方法报错
            content_str = str(content)
            content_lower = content_str.lower() if self.ignore_case else content_str
            
            # 完全匹配
            if content_lower == text:
                return choice
            
            # 包含匹配（需要较高的相似度）
            if len(content_lower) > 3 and content_lower in text:
                # 检查是否是主要内容
                if len(content_lower) / len(text) > 0.5:
                    return choice
        
        return None
    
    def _extract_by_sentence_transformer(self, pred_text: str, choices: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        使用 Sentence Transformer 语义相似度匹配选项
        
        优化策略：
        1. 先尝试用规则提取字母（如 "C" -> 选项C的内容）
        2. 如果提取到字母，用该字母对应的选项内容验证相似度
        3. 如果没有提取到字母，再用 sentence_transformer 直接匹配
        4. 置信度阈值 70，低于此值返回 None 让 LLM 回退
        
        Returns:
            Dict with matched_letter, matched_choice, confidence, or None if failed
        """
        if not self.sentence_transformer or not choices:
            return None
        
        try:
            # 将选项转换为列表（保持顺序）
            # 确保所有选项内容都是字符串类型，避免 float/int 导致后续处理报错
            choice_letters = list(choices.keys())
            choice_contents = [str(choices[letter]) if choices[letter] is not None else '' for letter in choice_letters]
            
            # 策略1：先尝试用规则提取字母
            rule_extracted_letter = self._extract_by_patterns(pred_text, choices)
            
            if rule_extracted_letter is not None:
                # 提取到字母，用该字母对应的选项内容来验证
                matched_letter = rule_extracted_letter.upper()
                matched_choice = str(choices.get(matched_letter, ""))
                
                # 计算预测文本与该选项内容的相似度（作为置信度验证）
                prediction_embedding = self.sentence_transformer.encode(pred_text, convert_to_tensor=True)
                choice_embedding = self.sentence_transformer.encode(matched_choice, convert_to_tensor=True)
                
                # 归一化
                prediction_embedding = F.normalize(prediction_embedding.unsqueeze(0), p=2, dim=1)
                choice_embedding = F.normalize(choice_embedding.unsqueeze(0), p=2, dim=1)
                
                # 计算余弦相似度
                confidence = (prediction_embedding @ choice_embedding.T).item() * 100
                
                # 规则提取成功，直接信任（但仍记录置信度）
                self.st_success_count += 1
                return {
                    'matched_letter': matched_letter,
                    'matched_choice': matched_choice,
                    'confidence': confidence,
                    'extraction_method': 'rule_then_verify'
                }
            
            # 策略2：规则提取失败，用 sentence_transformer 匹配所有选项
            prediction_embedding = self.sentence_transformer.encode(pred_text, convert_to_tensor=True)
            choice_embeddings = self.sentence_transformer.encode(choice_contents, convert_to_tensor=True)
            
            # 归一化（用于余弦相似度）
            prediction_embedding = F.normalize(prediction_embedding.unsqueeze(0), p=2, dim=1)
            choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)
            
            # 计算余弦相似度
            scores = (prediction_embedding @ choice_embeddings.T) * 100
            scores = scores.squeeze()
            
            # 选择相似度最高的选项
            best_choice_idx = torch.argmax(scores).item()
            matched_letter = choice_letters[best_choice_idx]
            matched_choice = choice_contents[best_choice_idx]
            confidence = torch.max(scores).item()
            
            # 置信度阈值 70，低于此值认为不可信
            if confidence >= 70:
                self.st_success_count += 1
                return {
                    'matched_letter': matched_letter.upper(),
                    'matched_choice': matched_choice,
                    'confidence': confidence,
                    'extraction_method': 'semantic_match'
                }
            else:
                # 置信度不足，返回 None 让 LLM 回退
                return None
                
        except Exception as e:
            print(f"Sentence Transformer extraction failed: {e}")
            return None
    
    def eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用规则提取选项答案
        按优先级尝试不同的提取方法：规则 -> Sentence Transformer -> (不再继续)
        """
        pred_text = str(prediction.get('prediction', ''))
        
        # 支持多种数据格式提取ground_truth
        ground_truth = None
        if 'ground_truth' in prediction:
            ground_truth = str(prediction['ground_truth']).upper()
        elif 'annotation' in prediction and prediction['annotation']:
            ground_truth = str(prediction['annotation'].get('reference', prediction['annotation'].get('gt_answer', ''))).upper()
        elif 'reference' in prediction:
            ground_truth = str(prediction['reference']).upper()
        else:
            raise ValueError('No ground truth found')
        
        # 构建选项字典
        choices = {}
        
        # 检查是否有 choices 字段（如 MMAU 数据集）
        # 支持最多 10 个选项 (A-J)
        ALL_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        if 'annotation' in prediction and 'choices' in prediction['annotation']:
            # 从 annotation 中获取 choices 列表
            choices_list = prediction['annotation']['choices']
            for i, choice_content in enumerate(choices_list):
                if i < len(ALL_OPTIONS):
                    opt = ALL_OPTIONS[i]
                    choices[opt] = choice_content
            
            # 将 gt_answer 转换为选项字母
            # 优先级：先检查 gt_answer 是否与某个选项内容匹配，再检查是否是选项字母
            # 这样可以处理 choices=["Yes", "No", "A", "B"], gt_answer="A" 的情况
            # （gt_answer="A" 应该对应选项 C，而不是选项 A）
            original_gt_answer = prediction['annotation'].get('gt_answer', '')
            content_matched = False
            
            # 标准化 ground_truth 用于模糊匹配（处理 "middle-aged" vs "middle aged" 等情况）
            gt_normalized = normalize_for_matching(ground_truth)
            
            # 首先尝试通过内容匹配（支持精确匹配和标准化匹配）
            for opt, content in choices.items():
                content_str = str(content) if content is not None else ''
                content_normalized = normalize_for_matching(content_str)
                # 匹配条件：完全匹配 或 原始值匹配 或 标准化后匹配
                if (content_str.upper() == ground_truth or 
                    content_str == str(original_gt_answer) or
                    content_normalized == gt_normalized):
                    ground_truth = opt
                    content_matched = True
                    break
            
            # 如果内容匹配失败，且 gt_answer 是选项字母，则直接使用
            if not content_matched and ground_truth in ALL_OPTIONS:
                # gt_answer 本身就是选项字母（A/B/C/D），直接使用
                pass
            elif not content_matched:
                # 既不是选项内容，也不是选项字母，无法处理
                pass
        else:
            # 原有逻辑：从 prediction 中直接获取选项
            for opt in ALL_OPTIONS:
                if opt in prediction:
                    choices[opt] = prediction[opt]
                elif f'option_{opt}' in prediction:
                    choices[opt] = prediction[f'option_{opt}']
        
        # 保存转换后的 ground_truth 到 prediction
        prediction['ground_truth'] = ground_truth
        
        # 按优先级尝试不同的提取方法
        extracted_answer = None
        eval_method = None
        
        # 1. 尝试模板匹配
        extracted_answer = self._extract_by_templates(pred_text, choices)
        if extracted_answer is not None:
            eval_method = 'template'
        
        # 2. 尝试模式匹配
        if extracted_answer is None:
            extracted_answer = self._extract_by_patterns(pred_text, choices)
            if extracted_answer is not None:
                eval_method = 'pattern'
        
        # 3. 尝试内容匹配
        if extracted_answer is None:
            extracted_answer = self._extract_by_content_matching(pred_text, choices)
            if extracted_answer is not None:
                eval_method = 'content'
        
        # 4. 尝试 Sentence Transformer 匹配（如果启用且前面的方法都失败）
        st_result = None
        if extracted_answer is None and self.use_sentence_transformer and self.sentence_transformer:
            st_result = self._extract_by_sentence_transformer(pred_text, choices)
            if st_result is not None:
                extracted_answer = st_result['matched_letter']
                eval_method = 'sentence_transformer'
                # 记录 sentence_transformer 的详细信息
                prediction['matched_choice'] = st_result['matched_choice']
                prediction['confidence'] = st_result['confidence']
                prediction['st_extraction_method'] = st_result.get('extraction_method', 'unknown')
            else:
                self.st_fallback_count += 1
        
        # 如果成功提取
        if extracted_answer is not None:
            is_correct = (extracted_answer == ground_truth)
            if is_correct:
                self.correct_samples += 1
        else:
            is_correct = False
            eval_method = 'failed'
            
        prediction['extracted_answer'] = extracted_answer
        prediction['score'] = 1.0 if is_correct else 0.0
        prediction['match'] = is_correct
        prediction['extract_fail'] = 1 if extracted_answer is None else 0
        prediction['eval_method'] = eval_method  # 记录使用的评估方法
        return prediction

    
    def llm_eval(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用LLM提取选项答案
        参考VLMEvalKit的build_prompt方法
        """
        if not self.llm_client:
            return None
        
        pred_text = str(prediction.get('prediction', ''))
        
        # 支持多种数据格式提取ground_truth
        ground_truth = None
        if 'ground_truth' in prediction:
            ground_truth = str(prediction['ground_truth']).upper()
        elif 'annotation' in prediction and prediction['annotation']:
            ground_truth = str(prediction['annotation'].get('reference', prediction['annotation'].get('gt_answer', ''))).upper()
        elif 'reference' in prediction:
            ground_truth = str(prediction['reference']).upper()
        else:
            raise ValueError('No ground truth found')
        
        # 支持多种数据格式提取question
        question = ''
        if 'question' in prediction:
            question = prediction['question']
        elif 'annotation' in prediction and prediction['annotation']:
            question = prediction['annotation'].get('prompt', prediction['annotation'].get('question', ''))
        
        # 构建选项字典
        choices = {}
        
        # 检查是否有 choices 字段（如 MMAU 数据集）
        # 支持最多 10 个选项 (A-J)
        ALL_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        if 'annotation' in prediction and 'choices' in prediction['annotation']:
            # 从 annotation 中获取 choices 列表
            choices_list = prediction['annotation']['choices']
            for i, choice_content in enumerate(choices_list):
                if i < len(ALL_OPTIONS):
                    opt = ALL_OPTIONS[i]
                    choices[opt] = choice_content
            
            # 将 gt_answer 转换为选项字母
            # 优先级：先检查 gt_answer 是否与某个选项内容匹配，再检查是否是选项字母
            # 这样可以处理 choices=["Yes", "No", "A", "B"], gt_answer="A" 的情况
            # （gt_answer="A" 应该对应选项 C，而不是选项 A）
            original_gt_answer = prediction['annotation'].get('gt_answer', '')
            content_matched = False
            
            # 标准化 ground_truth 用于模糊匹配（处理 "middle-aged" vs "middle aged" 等情况）
            gt_normalized = normalize_for_matching(ground_truth)
            
            # 首先尝试通过内容匹配（支持精确匹配和标准化匹配）
            for opt, content in choices.items():
                content_str = str(content) if content is not None else ''
                content_normalized = normalize_for_matching(content_str)
                # 匹配条件：完全匹配 或 原始值匹配 或 标准化后匹配
                if (content_str.upper() == ground_truth or 
                    content_str == str(original_gt_answer) or
                    content_normalized == gt_normalized):
                    ground_truth = opt
                    content_matched = True
                    break
            
            # 如果内容匹配失败，且 gt_answer 是选项字母，则直接使用
            if not content_matched and ground_truth in ALL_OPTIONS:
                # gt_answer 本身就是选项字母（A/B/C/D），直接使用
                pass
            elif not content_matched:
                # 既不是选项内容，也不是选项字母，无法处理
                pass
        else:
            # 原有逻辑：从 prediction 中直接获取选项
            for opt in ALL_OPTIONS:
                if opt in prediction:
                    choices[opt] = prediction[opt]
                elif f'option_{opt}' in prediction:
                    choices[opt] = prediction[f'option_{opt}']
        
        # 保存转换后的 ground_truth 到 prediction
        prediction['ground_truth'] = ground_truth
        
        # 构建选项字符串
        option_str = "\n".join([f"{k}. {v}" for k, v in choices.items()])
        
        # 使用预定义的prompt模板
        prompt = MQA_EXTRACTION_PROMPT.format(
            choices=", ".join(choices.keys()),
            question=question,
            options=option_str,
            prediction=pred_text
        )
        
        try:
            response = self.llm_client.get_eval(content=prompt, max_tokens=10, temperature=0.0, model_name=APIModelName.GPT_4O_MINI)
            prediction['llm_response'] = response
            response = response.strip().upper()
            
            # 提取答案
            extracted_answer = response
            
            if extracted_answer and extracted_answer != 'Z':
                is_correct = (extracted_answer == ground_truth)
                if is_correct:
                    self.correct_samples += 1
                
                prediction['extracted_answer'] = extracted_answer
                prediction['score'] = 1.0 if is_correct else 0.0
                prediction['match'] = is_correct
                prediction['extract_fail'] = 0
                return prediction
            else:
                prediction['extracted_answer'] = ''
                prediction['score'] = 0.0
                prediction['match'] = False
                prediction['extract_fail'] = 1
                return prediction
        except Exception as e:
            print(f"Error calling LLM: {e}")
        
        
    def summary(self) -> Tuple[str, float]:
        """生成评估摘要"""
        if self.total_samples == 0:
            accuracy = 0.0
        else:
            accuracy = self.correct_samples / self.total_samples
        
        stats = self.get_eval_stats()
        
        report = f"MQA Evaluation Report\n"
        report += f"{'=' * 50}\n"
        report += f"Total Samples: {self.total_samples}\n"
        report += f"Correct: {self.correct_samples}\n"
        report += f"Accuracy: {accuracy:.2%}\n"
        report += f"\nEvaluation Method Stats:\n"
        report += f"  Rule-based: {stats['rule_eval_count']} ({stats['rule_eval_rate']:.1%})\n"
        report += f"  LLM-based: {stats['llm_eval_count']} ({stats['llm_eval_rate']:.1%})\n"
        report += f"  Failed: {stats['failed_count']} ({stats['failed_rate']:.1%})\n"
        
        # Sentence Transformer 统计（如果启用）
        if self.use_sentence_transformer:
            total_attempts = self.st_success_count + self.st_fallback_count
            report += f"\nSentence Transformer Statistics (after rule-based methods failed):\n"
            report += f"  Sentence Transformer success: {self.st_success_count}"
            if total_attempts > 0:
                report += f" ({self.st_success_count/total_attempts:.1%})\n"
            else:
                report += "\n"
            report += f"  Sentence Transformer skipped/failed: {self.st_fallback_count}"
            if total_attempts > 0:
                report += f" ({self.st_fallback_count/total_attempts:.1%})\n"
            else:
                report += "\n"
        
        # 添加 task 统计信息（使用基类的方法）
        report += self.get_task_stats_report()
        
        # 添加分组统计（使用基类的方法）
        report += self.get_group_stats_report()
        
        return report, accuracy
    
    def __del__(self):
        """释放 Sentence Transformer 模型内存"""
        if self.sentence_transformer is not None:
            del self.sentence_transformer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    # 测试MQAEvaluator
    print("=== 测试 MQAEvaluator ===")
    
    evaluator = MQAEvaluator(use_llm_fallback=True)
    
    # 测试数据
    test_cases = [
        {
            'question': 'What is the capital of France?\nOptions: A. London B. Berlin C. Paris D. Madrid',
            'choices': ['London', 'Berlin', 'Paris', 'Madrid'],
            'prediction': 'C',
            'ground_truth': 'C'
        },
        {
            'question': 'Which of these are programming languages?\nOptions: A. Python B. Java C. HTML D. CSS',
            'choices': ['Python', 'Java', 'HTML', 'CSS'],
            'prediction': 'A,B,C,D',
            'ground_truth': 'A'  # 多选题
        },
        {
            'question': 'What is 2+2?\nOptions: A. 3 B. 4 C. 5 D. 6',
            'choices': ['3', '4', '5', '6'],
            'prediction': 'The answer is 4, so B',
            'ground_truth': 'B'
        }
    ]
    
    # 评估每个测试用例
    for i, case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"问题: {case['question']}")
        print(f"选项: {case['choices']}")
        print(f"预测: {case['prediction']}")
        print(f"答案: {case['ground_truth']}")
        
        result = evaluator.evaluate_single(case)
        print(f"得分: {result['score']}")
        print(f"评估方法: {result.get('eval_method', 'unknown')}")
        if 'extracted_answer' in result:
            print(f"提取的答案: {result['extracted_answer']}")
    
    # 生成报告
    report, accuracy = evaluator.summary()
    print(f"\n{report}")
