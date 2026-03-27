import json

from o_e_Kit.utils.metrics.wer_eval import WER_Eval
from o_e_Kit.utils.metrics.mcq_eval import MCQ_Eval
from o_e_Kit.utils.metrics.streaming_bench_eval import StreamingBenchEval
from o_e_Kit.utils.metrics.omni_llm_eval import OmniLLMEvaluator
from o_e_Kit.utils.metrics.evaluator_refqa import RefQAEvaluator
from o_e_Kit.utils.metrics.evaluator_mqa import MQAEvaluator
from o_e_Kit.utils.metrics.evaluator_openqa import OpenQAEvaluator
from o_e_Kit.utils.metrics.evaluator_safety import SafetyEvaluator
from o_e_Kit.utils.metrics.evaluator_instruction_following import InstructionFollowingEvaluator
from o_e_Kit.utils.metrics.evaluator_caption import CaptionEvaluator
from o_e_Kit.utils.metrics.evaluator_livesports3k_llm_judge import LiveSports3KLLMJudgeEvaluator
from o_e_Kit.utils.metrics.evaluator_ovavel import OVAVELEvaluator

def evaluate_dataset(dataset_name: str, answer_file_path: str) -> float:
    with open(answer_file_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
        
    if isinstance(predictions, dict) and 'predictions' in predictions:
        predictions = predictions['predictions']
    else:
        # 兼容旧格式
        predictions = predictions
    scored_predictions = []

    # 如果存在通过 API 调用得到的结果（例如 Gemini），我们允许在 predictions 中
    # 标记 api_ok=False 的样本，这类样本会被排除在评估之外，避免网络问题影响准确率。
    if isinstance(predictions, list):
        total_before = len(predictions)
        predictions = [
            p for p in predictions
            if not (isinstance(p, dict) and p.get("other", {}).get("api_ok") is False)
        ]
        skipped = total_before - len(predictions)
        if skipped > 0:
            print(
                f"⚠️ 跳过了 {skipped}/{total_before} 条带有 api_ok=False 标记的样本，"
                f"这些样本的 API 调用失败，不计入 {dataset_name} 的评估分母。"
            )
    
    # 英文ASR数据集 - 使用 WER
    english_asr_datasets = [
        "gigaspeech_test", "librispeech_test_clean", "librispeech_test_other",
        "librispeech_dev_clean", "librispeech_dev_other", "commonvoice_en",
        "voxpopuli_en", "fleurs_en", "peoples_speech_test", "tedlium3_test"
    ]
    
    # 中文ASR数据集 - 使用 CER
    chinese_asr_datasets = [
        "wenetspeech_test_net", "wenetspeech_test_meeting", "commonvoice_zh",
        "commonvoice_yue", "aishell1_test", "aishell2_test",
        "kespeech_test", "fleurs_zh"
    ]
    
    # 法语ASR数据集 - 使用 WER
    french_asr_datasets = ["commonvoice_fr"]
    

    refqa_datasets = ["voicebench_sdqa", "voicebench_bbh", "audio_web_questions", "audio_trivia_qa"]
    mqa_datasets = ["voicebench_mmsu", "voicebench_openbookqa", "voice_cmmlu", "mmau_test_mini", "mmsu_bench", "mmar_bench"]
    openqa_datasets = ["voicebench_alpacaeval", "voicebench_alpacaeval_full", "voicebench_commoneval", "voicebench_wildvoice"]
    openqa_safety_datasets = ["voicebench_advbench"]
    instruction_following_datasets = ["voicebench_ifeval"]
    
    # Audio/Video Caption datasets
    caption_datasets = ["audiocaps_test", "clothocaption_test", "wavcaps_audioset_sl", 
                       "wavcaps_freesound", "wavcaps_soundbible", 
                       "covost2_zh_en", "covost2_en_zh"
                       ]
    
    # Audio Classification datasets
    classification_datasets = ["vocalsound", "meld"]
    
    if dataset_name in english_asr_datasets:
        metric = 'wer'
        evaluator = WER_Eval(lang='en', metric=metric)
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
    elif dataset_name in chinese_asr_datasets:
        metric = 'cer'
        evaluator = WER_Eval(lang='zh', metric=metric)
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
    elif dataset_name in french_asr_datasets:
        metric = 'wer'
        evaluator = WER_Eval(lang='fr', metric=metric)
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
    elif dataset_name in refqa_datasets:
        # SD-QA 使用 RefQA 评估器（有参考答案的问答）
        # BBH 是 Yes/No 问答，使用 RefQA 评估器
        metric = 'refqa'
        evaluator = RefQAEvaluator(use_llm_fallback=True, strict_matching=False)
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name in mqa_datasets:
        # 选择题数据集使用 MQA 评估器
        # use_sentence_transformer=True: 启用 Sentence Transformer 语义匹配（在规则匹配失败后使用）
        # use_sentence_transformer=True: 只使用规则匹配和LLM后备
        metric = 'mqa'
        evaluator = MQAEvaluator(
            use_llm_fallback=True,            # 启用LLM作为最后的后备方法
            use_sentence_transformer=True    # 可选：启用 Sentence Transformer 语义匹配（推荐设为True）
        )
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name in openqa_datasets:
        # 开放式问答 - 评估答案质量（1-5分制）
        metric = 'openqa'
        evaluator = OpenQAEvaluator(use_llm_fallback=True)
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name in openqa_safety_datasets:
        # 对抗性测试，检测拒绝回答（0-1分制）
        metric = 'openqa_safety'
        evaluator = SafetyEvaluator()
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name in instruction_following_datasets:
        # IFEval 需要特殊处理
        metric = 'instruction_following'
        evaluator = InstructionFollowingEvaluator()
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
    
    elif dataset_name in caption_datasets:
        # Audio Caption 评估 - 使用 BLEU, METEOR, CIDEr, SPIDEr 等指标
        metric = 'caption'
        
        # 根据数据集类型选择 BLEU 计算方法和目标语言
        # 翻译任务使用 sacrebleu，caption 任务使用 pycocoevalcap
        if dataset_name == "covost2_zh_en":
            # 中译英：目标语言是英语
            evaluator = CaptionEvaluator(use_llm_fallback=False, bleu_method="sacrebleu", target_lang="en")
        elif dataset_name == "covost2_en_zh":
            # 英译中：目标语言是中文
            evaluator = CaptionEvaluator(use_llm_fallback=False, bleu_method="sacrebleu", target_lang="zh")
        else:
            # 其他 caption 数据集使用 pycocoevalcap（默认）
            evaluator = CaptionEvaluator(use_llm_fallback=False, bleu_method="pycocoevalcap")
        
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name in classification_datasets:
        # Audio Classification 评估 - 使用精确匹配
        metric = 'classification'
        evaluator = RefQAEvaluator(use_llm_fallback=True, strict_matching=False)
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name == "OVOBench":
        metric = 'MCQ'
        evaluator = MCQ_Eval(metric=metric)
        evaluator.scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
    elif dataset_name.startswith("StreamingBench"):
        # 提取任务类型
        if "REAL" in dataset_name:
            task_type = "real"
        elif "OMNI" in dataset_name:
            task_type = "omni"
        elif "SQA" in dataset_name:
            task_type = "sqa"
        elif "PROACTIVE" in dataset_name:
            task_type = "proactive"
        else:
            task_type = "real"  # 默认
        
        metric = 'StreamingBench'
        
        evaluator = StreamingBenchEval(task_type=task_type)
            
        scored_predictions = evaluator.evaluate(predictions)
        summary_result = evaluator.summary()
        final_score = summary_result['overall_accuracy']
        
        # 打印详细报告
        evaluator.print_summary()
        report = f"StreamingBench {task_type.upper()} Task Evaluation Report\n"
        report += f"Overall Accuracy: {final_score:.2%}\n"
        report += f"Total Questions: {summary_result['total_questions']}\n"
        report += f"Correct Answers: {summary_result['correct_answers']}\n"
    elif dataset_name == "livesports3k_cc":
        # LiveSports-3K CC 使用 LLM Judge 评估 (基于 LiveCC 官方实现)
        # 参考: https://github.com/showlab/livecc/blob/main/evaluation/livesports3kcc/llm_judge.py
        metric = 'llm_judge'
        evaluator = LiveSports3KLLMJudgeEvaluator(
            baseline_id="GPT-4o",
            model_id="Model",
            max_workers=8,
            group_by_fields=['class']  # 按运动类别分组统计
        )
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name in ["VisionCap", "VisionCap_Offline", "VisionCap_Offline2", "OmniCap", "OmniCap_Offline", "OmniCap_Offline2", "LiveCC", "AVEvent"]:
        # Duplex数据集使用LLM评估
        metric = 'omni_llm'
        evaluator = OmniLLMEvaluator()
        
        # 对于offline2数据集使用简单评估方法
        if dataset_name in ["VisionCap_Offline2", "OmniCap_Offline2"]:
            scored_predictions, report, final_score = evaluator.evaluate_simple_batch(predictions)
        else:
            # 使用完整的评估方法
            scored_predictions, report, final_score = evaluator.batch_evaluate(predictions)
    
    # ===== Omni 多模态数据集评估（统一 JSONL，MCQ/MQA）=====
    elif dataset_name in [
        "daily_omni",
        "omnibench",
        "worldsense",
        "av_odyssey",
        "videomme",
        "videomme_short",
        "unobench_mc",
        "ovobench",
        "video_holmes",
        "avut_benchmark_human",
        "avut_benchmark_gemini",
        "streamingbench_real",
        "streamingbench_omni_fix",
        "streamingbench_sqa",
        "jointavbench",
        "futureomni",
        "avmeme_full",
        "avmeme_main",
    ]:
        # MCQ 评估，根据 dataset_name 自动选择分组字段
        metric = 'mqa'
        
        # 各数据集分组字段（用于按任务/能力/时长等维度统计）
        group_fields = {
            'daily_omni': ['qa_type', 'content_parent_category', 'video_category', 'video_duration'],
            'omnibench': ['task_type', 'audio_type'],
            'worldsense': ['task_type', 'duration'],
            'av_odyssey': ['subfield', 'question_type_id', 'data_type'],
            # Video-MME：按任务类型 / 时长类别分组（short / medium / long）
            'videomme': ['task_type', 'duration'],
            # Video-MME Short：依然保留 task_type/duration 字段，便于与完整集对齐比较
            'videomme_short': ['task_type', 'duration'],
            'unobench_mc': ['ability', 'task', 'audio_type'],
            # OVO-Bench：按任务类型 (EPM/ASI/... 等) 分组
            'ovobench': ['task'],
            # Video-Holmes：按问题类型 / 视频时长分组
            'video_holmes': ['question_type', 'video_duration'],
            # AVUT-Benchmark：按任务类型 / 视频类型分组
            'avut_benchmark_human': ['task_type', 'video_type'],
            'avut_benchmark_gemini': ['task_type', 'video_type'],
            # StreamingBench：按任务类型 / 能力 / 视频类别分组
            'streamingbench_real': ['task_type', 'required_ability', 'video_categories'],
            'streamingbench_omni_fix': ['task_type', 'required_ability', 'video_categories'],
            'streamingbench_sqa': ['task_type', 'required_ability', 'video_categories'],
            # JointAVBench：按任务类型分组（15种任务：STL/SOER/SPER/...）
            'jointavbench': ['task'],
            # FutureOmni：按视频领域/音频类型/预测模式分组
            'futureomni': ['video_domain', 'audio_type', 'forecasting_pattern'],
            # AVMeme-Exam：按声音类别/问题类型/语言分组
            'avmeme_full': ['category', 'question_type', 'language'],
            'avmeme_main': ['category', 'question_type', 'language'],
        }
        
        evaluator = MQAEvaluator(
            use_llm_fallback=True, 
            use_sentence_transformer=True,
            group_by_fields=group_fields.get(dataset_name, [])
        )
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    elif dataset_name == "unobench":
        # UNO-Bench: MCQ + Open-Ended 混合评估
        metric = 'mqa_open'
        evaluator = RefQAEvaluator(
            use_llm_fallback=True,
            group_by_fields=['subset_name', 'score_type']
        )
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
    
    elif dataset_name == "ovavel":
        # OV-AVEL: Open-Vocabulary Audio-Visual Event Localization
        # 评估指标：帧级准确率、片段级 F1、事件级 F1
        # 参考论文: https://arxiv.org/pdf/2411.11278
        metric = 'ovavel'
        evaluator = OVAVELEvaluator(
            fps=1.0,                    # 1帧/秒，10秒视频 = 10帧
            iou_threshold=0.5,          # 片段匹配的 IoU 阈值
            group_by_fields=['event_category', 'cls_type']  # 按事件类别和开放/基础类型分组
        )
        scored_predictions = evaluator.evaluate(predictions)
        report, final_score = evaluator.summary()
        
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Please add an evaluation flow for it in 'evaluate_dataset' function."
        )

    print(report)
    report_path = answer_file_path.replace('.json', f'_{metric}.txt')
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    
    answer_model_method_path = answer_file_path.replace('.json', f'_{metric}.json')
    with open(answer_model_method_path, "w", encoding='utf-8') as f:
        json.dump(scored_predictions, f, indent=4, ensure_ascii=False)

    return final_score

