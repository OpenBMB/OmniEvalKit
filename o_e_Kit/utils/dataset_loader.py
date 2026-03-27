"""
数据集加载模块
负责加载各种评估数据集
"""

import torch
from o_e_Kit.datasets.audio_datasets import AudioEvalDataset
from o_e_Kit.datasets.omni_datasets import omni_datasetWoann, OmniEvalDataset
from o_e_Kit.utils.args.dataset_args import DATASET_REGISTRY


def build_itembuilder_args(args):
    """构建duplex数据集所需的itembuilder_args"""
    return {
        'img_size': args.duplex_img_size,
        'img_aug': args.duplex_img_aug,
        'use_adaptive_slice': args.duplex_use_adaptive_slice,
        'total_max_length': args.duplex_total_max_length,
        'insert_listen_speak': args.duplex_insert_listen_speak,
        'audio_pool_step': args.duplex_audio_pool_step,
        'audio_chunk_length': args.duplex_audio_chunk_length,
        'streaming_text_reserved_len': args.duplex_streaming_text_reserved_len,
        'insert_vad': args.duplex_insert_vad,
        'audio_normalize': args.duplex_audio_normalize,
        'tokenizer_path': args.tokenizer_path,
        'stable_audio_token_interval_for_duplex': args.duplex_stable_audio_token_interval,
    }


def load_dataset(args, dataset_name, current_task=None):
    """加载指定的数据集
    
    Args:
        args: 参数对象，包含max_sample_num等配置
        dataset_name: 数据集名称
        current_task: 当前任务（用于StreamingBench等多任务数据集）
    
    Returns:
        dataset: 加载的数据集，如果指定了max_sample_num且有效，会返回子集
    """
    print(f"Loading dataset: {dataset_name}...")
    
    if dataset_name == "gigaspeech_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.gigaspeech_test_data_prefix_dir,
            annotation_path=args.gigaspeech_test_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "wenetspeech_test_net":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.wenetspeech_test_net_data_prefix_dir,
            annotation_path=args.wenetspeech_test_net_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "wenetspeech_test_meeting":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.wenetspeech_test_meeting_data_prefix_dir,
            annotation_path=args.wenetspeech_test_meeting_annotation_path,
            dataset_name=dataset_name
        )
        
    # LibriSpeech系列数据集
    elif dataset_name == "librispeech_test_clean":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.librispeech_test_clean_data_prefix_dir,
            annotation_path=args.librispeech_test_clean_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "librispeech_test_other":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.librispeech_test_other_data_prefix_dir,
            annotation_path=args.librispeech_test_other_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "librispeech_dev_clean":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.librispeech_dev_clean_data_prefix_dir,
            annotation_path=args.librispeech_dev_clean_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "librispeech_dev_other":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.librispeech_dev_other_data_prefix_dir,
            annotation_path=args.librispeech_dev_other_annotation_path,
            dataset_name=dataset_name
        )
        
    # CommonVoice系列数据集
    elif dataset_name == "commonvoice_en":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.commonvoice_en_data_prefix_dir,
            annotation_path=args.commonvoice_en_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "commonvoice_zh":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.commonvoice_zh_data_prefix_dir,
            annotation_path=args.commonvoice_zh_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "commonvoice_yue":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.commonvoice_yue_data_prefix_dir,
            annotation_path=args.commonvoice_yue_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "commonvoice_fr":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.commonvoice_fr_data_prefix_dir,
            annotation_path=args.commonvoice_fr_annotation_path,
            dataset_name=dataset_name
        )
        
    # AISHELL-1数据集
    elif dataset_name == "aishell1_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.aishell1_test_data_prefix_dir,
            annotation_path=args.aishell1_test_annotation_path,
            dataset_name=dataset_name
        )
        
    # AISHELL-2数据集
    elif dataset_name == "aishell2_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.aishell2_test_data_prefix_dir,
            annotation_path=args.aishell2_test_annotation_path,
            dataset_name=dataset_name
        )
    
    # KeSpeech数据集
    elif dataset_name == "kespeech_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.kespeech_test_data_prefix_dir,
            annotation_path=args.kespeech_test_annotation_path,
            dataset_name=dataset_name
        )
        
    # VoxPopuli数据集
    elif dataset_name == "voxpopuli_en":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voxpopuli_en_data_prefix_dir,
            annotation_path=args.voxpopuli_en_annotation_path,
            dataset_name=dataset_name
        )
        
    # FLEURS系列数据集
    elif dataset_name == "fleurs_zh":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.fleurs_zh_data_prefix_dir,
            annotation_path=args.fleurs_zh_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "fleurs_en":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.fleurs_en_data_prefix_dir,
            annotation_path=args.fleurs_en_annotation_path,
            dataset_name=dataset_name
        )
        
    # People's Speech数据集
    elif dataset_name == "peoples_speech_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.peoples_speech_test_data_prefix_dir,
            annotation_path=args.peoples_speech_test_annotation_path,
            dataset_name=dataset_name
        )
        
    # TED-LIUM v3数据集
    elif dataset_name == "tedlium3_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.tedlium3_test_data_prefix_dir,
            annotation_path=args.tedlium3_test_annotation_path,
            dataset_name=dataset_name
        )
        
    # VoiceBench QA系列数据集
    elif dataset_name == "voicebench_alpacaeval":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_alpacaeval_data_prefix_dir,
            annotation_path=args.voicebench_alpacaeval_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_alpacaeval_full":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_alpacaeval_full_data_prefix_dir,
            annotation_path=args.voicebench_alpacaeval_full_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_bbh":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_bbh_data_prefix_dir,
            annotation_path=args.voicebench_bbh_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_mmsu":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_mmsu_data_prefix_dir,
            annotation_path=args.voicebench_mmsu_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_openbookqa":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_openbookqa_data_prefix_dir,
            annotation_path=args.voicebench_openbookqa_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_advbench":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_advbench_data_prefix_dir,
            annotation_path=args.voicebench_advbench_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_commoneval":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_commoneval_data_prefix_dir,
            annotation_path=args.voicebench_commoneval_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_ifeval":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_ifeval_data_prefix_dir,
            annotation_path=args.voicebench_ifeval_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_sdqa":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_sdqa_data_prefix_dir,
            annotation_path=args.voicebench_sdqa_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voice_cmmlu":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voice_cmmlu_data_prefix_dir,
            annotation_path=args.voice_cmmlu_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "audio_trivia_qa":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.audio_trivia_qa_data_prefix_dir,
            annotation_path=args.audio_trivia_qa_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "audio_web_questions":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.audio_web_questions_data_prefix_dir,
            annotation_path=args.audio_web_questions_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "voicebench_wildvoice":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.voicebench_wildvoice_data_prefix_dir,
            annotation_path=args.voicebench_wildvoice_annotation_path,
            dataset_name=dataset_name
        )
        
    # MMAU 多任务音频理解数据集
    elif dataset_name == "mmau_test_mini":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.mmau_test_mini_data_prefix_dir,
            annotation_path=args.mmau_test_mini_annotation_path,
            dataset_name=dataset_name
        )
    
    # MMSU 多任务语音理解数据集
    elif dataset_name == "mmsu_bench":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.mmsu_bench_data_prefix_dir,
            annotation_path=args.mmsu_bench_annotation_path,
            dataset_name=dataset_name
        )
    
    # MMAR 多模态音频推理数据集
    elif dataset_name == "mmar_bench":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.mmar_bench_data_prefix_dir,
            annotation_path=args.mmar_bench_annotation_path,
            dataset_name=dataset_name
        )
    
    # Audio Caption系列数据集
    elif dataset_name == "audiocaps_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.audiocaps_test_data_prefix_dir,
            annotation_path=args.audiocaps_test_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "clothocaption_test":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.clothocaption_test_data_prefix_dir,
            annotation_path=args.clothocaption_test_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "wavcaps_audioset_sl":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.wavcaps_audioset_sl_data_prefix_dir,
            annotation_path=args.wavcaps_audioset_sl_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "wavcaps_freesound":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.wavcaps_freesound_data_prefix_dir,
            annotation_path=args.wavcaps_freesound_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "wavcaps_soundbible":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.wavcaps_soundbible_data_prefix_dir,
            annotation_path=args.wavcaps_soundbible_annotation_path,
            dataset_name=dataset_name
        )
        
    # Audio Classification datasets
    elif dataset_name == "vocalsound":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.vocalsound_data_prefix_dir,
            annotation_path=args.vocalsound_annotation_path,
            dataset_name=dataset_name
        )
        
    # MELD emotion recognition dataset
    elif dataset_name == "meld":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.meld_data_prefix_dir,
            annotation_path=args.meld_annotation_path,
            dataset_name=dataset_name
        )
    
    # CoVoST2 AST datasets
    elif dataset_name == "covost2_zh_en":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.covost2_zh_en_data_prefix_dir,
            annotation_path=args.covost2_zh_en_annotation_path,
            dataset_name=dataset_name
        )
    elif dataset_name == "covost2_en_zh":
        dataset = AudioEvalDataset(
            data_prefix_dir=args.covost2_en_zh_data_prefix_dir,
            annotation_path=args.covost2_en_zh_annotation_path,
            dataset_name=dataset_name
        )
        
    elif dataset_name == "livesports3k_cc":
        dataset = OmniEvalDataset(
            annotation_path=args.livesports3k_cc_annotation_path,
            data_prefix_dir=args.livesports3k_cc_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "jointavbench":
        # JointAVBench：音视频联合理解 MCQ
        dataset = OmniEvalDataset(
            annotation_path=args.jointavbench_annotation_path,
            data_prefix_dir=args.jointavbench_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "ovavel":
        # OV-AVEL: Open-Vocabulary Audio-Visual Event Localization
        dataset = OmniEvalDataset(
            annotation_path=args.ovavel_annotation_path,
            data_prefix_dir=args.ovavel_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "ovobench":
        # OVO-Bench（统一 Omni JSONL）：离线视频 QA/MCQ
        dataset = OmniEvalDataset(
            annotation_path=args.ovobench_annotation_path,
            data_prefix_dir=args.ovobench_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "streamingbench_real":
        # StreamingBench-Real（统一 Omni JSONL）：离线视频 QA/MCQ（仅视觉）
        dataset = OmniEvalDataset(
            annotation_path=args.streamingbench_real_annotation_path,
            data_prefix_dir=args.streamingbench_real_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "streamingbench_omni":
        # StreamingBench-Omni（统一 Omni JSONL）：离线多模态 QA/MCQ（音频+视频）
        dataset = OmniEvalDataset(
            annotation_path=args.streamingbench_omni_annotation_path,
            data_prefix_dir=args.streamingbench_omni_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "streamingbench_omni_fix":
        # StreamingBench-Omni-Fix：筛选后的子集（仅 Emotion Recognition, Scene Understanding, Multimodal Alignment, Source Discrimination）
        dataset = OmniEvalDataset(
            annotation_path=args.streamingbench_omni_fix_annotation_path,
            data_prefix_dir=args.streamingbench_omni_fix_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "streamingbench_sqa":
        # StreamingBench-SQA（统一 Omni JSONL）：离线顺序多轮 QA
        dataset = OmniEvalDataset(
            annotation_path=args.streamingbench_sqa_annotation_path,
            data_prefix_dir=args.streamingbench_sqa_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "videomme":
        # Video-MME: 使用 OmniEvalDataset，不自动提取音频（视频本身包含音频轨道）
        dataset = OmniEvalDataset(
            annotation_path=args.videomme_annotation_path,
            data_prefix_dir=args.videomme_data_prefix_dir,
            dataset_name=dataset_name,
            auto_extract_audio=False
        )
    elif dataset_name == "videomme_short":
        # Video-MME Short: 使用离线生成的 short 子集 JSONL（仅包含 duration=short 样本）
        dataset = OmniEvalDataset(
            annotation_path=args.videomme_short_annotation_path,
            data_prefix_dir=args.videomme_short_data_prefix_dir,
            dataset_name=dataset_name,
            auto_extract_audio=False
        )
    elif dataset_name == "daily_omni":
        dataset = OmniEvalDataset(
            annotation_path=args.daily_omni_annotation_path,
            data_prefix_dir=args.daily_omni_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "omnibench":
        dataset = OmniEvalDataset(
            annotation_path=args.omnibench_annotation_path,
            data_prefix_dir=args.omnibench_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "unobench":
        dataset = OmniEvalDataset(
            annotation_path=args.unobench_annotation_path,
            data_prefix_dir=args.unobench_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "unobench_mc":
        dataset = OmniEvalDataset(
            annotation_path=args.unobench_mc_annotation_path,
            data_prefix_dir=args.unobench_mc_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "worldsense":
        dataset = OmniEvalDataset(
            annotation_path=args.worldsense_annotation_path,
            data_prefix_dir=args.worldsense_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "av_odyssey":
        dataset = OmniEvalDataset(
            annotation_path=args.av_odyssey_annotation_path,
            data_prefix_dir=args.av_odyssey_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "video_holmes":
        dataset = OmniEvalDataset(
            annotation_path=args.video_holmes_annotation_path,
            data_prefix_dir=args.video_holmes_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "avut_benchmark_human":
        dataset = OmniEvalDataset(
            annotation_path=args.avut_benchmark_human_annotation_path,
            data_prefix_dir=args.avut_benchmark_human_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "avut_benchmark_gemini":
        dataset = OmniEvalDataset(
            annotation_path=args.avut_benchmark_gemini_annotation_path,
            data_prefix_dir=args.avut_benchmark_gemini_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "futureomni":
        # FutureOmni: 未来预测评测基准
        dataset = OmniEvalDataset(
            annotation_path=args.futureomni_annotation_path,
            data_prefix_dir=args.futureomni_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "avmeme_full":
        # AVMeme-Exam Full: 音视频 Meme 理解评测 (1032 samples)
        # visual_cheat=true 的样本只有 WavPath，visual_cheat=false 的有 VideoPath
        dataset = OmniEvalDataset(
            annotation_path=args.avmeme_full_annotation_path,
            data_prefix_dir=args.avmeme_full_data_prefix_dir,
            dataset_name=dataset_name
        )
    elif dataset_name == "avmeme_main":
        # AVMeme-Exam Main: 去除 text_cheat 后的数据集 (846 samples)
        dataset = OmniEvalDataset(
            annotation_path=args.avmeme_main_annotation_path,
            data_prefix_dir=args.avmeme_main_data_prefix_dir,
            dataset_name=dataset_name
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 处理 max_sample_num 参数
    # 1. 检查是否设置了 max_sample_num
    # 2. 如果是 None 或 -1，不做限制
    # 3. 如果数据集本身没有那么长，不做限制
    if hasattr(args, 'max_sample_num') and args.max_sample_num is not None and args.max_sample_num > 0:
        if len(dataset) > args.max_sample_num:
            # 检查是否需要 shuffle (从全量随机采样)
            if getattr(args, 'shuffle', False):
                import random
                seed = getattr(args, 'shuffle_seed', 42)
                random.seed(seed)
                indices = random.sample(range(len(dataset)), args.max_sample_num)
                print(f"Randomly sampling {args.max_sample_num} from {len(dataset)} samples (seed={seed}).")
            # 检查是否需要 shuffle_after_limit (先取前N条，再打乱)
            elif getattr(args, 'shuffle_after_limit', False):
                import random
                seed = getattr(args, 'shuffle_seed', 42)
                random.seed(seed)
                indices = list(range(args.max_sample_num))
                random.shuffle(indices)
                print(f"Taking first {args.max_sample_num} samples, then shuffling (seed={seed}).")
            else:
                indices = list(range(args.max_sample_num))
            print(f"Limiting dataset from {len(dataset)} to {args.max_sample_num} samples.")
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            print(f"Dataset has {len(dataset)} samples (max_sample_num={args.max_sample_num}, no limiting needed).")
            # 即使不需要限制，也检查 shuffle_after_limit
            if getattr(args, 'shuffle_after_limit', False):
                import random
                seed = getattr(args, 'shuffle_seed', 42)
                random.seed(seed)
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                dataset = torch.utils.data.Subset(dataset, indices)
                print(f"  (已随机打乱，种子: {seed})")
    else:
        print(f"Data provider loaded for {dataset_name}, {len(dataset)} samples.")
        
        # 处理 shuffle（当没有 max_sample_num 限制时）
        if getattr(args, 'shuffle', False) or getattr(args, 'shuffle_after_limit', False):
            import random
            seed = getattr(args, 'shuffle_seed', 42)
            random.seed(seed)
            # 生成随机索引
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            # 使用 Subset 来实现 shuffle
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"  (已随机打乱，种子: {seed})")
    
    # 多路评测：在单个 job 内手动按照 route_idx / route_num 切分数据子集
    route_num = getattr(args, "route_num", 1)
    route_idx = getattr(args, "route_idx", 0)
    if route_num > 1:
        if route_idx < 0 or route_idx >= route_num:
            raise ValueError(f"route_idx 必须在 [0, route_num) 范围内，当前: {route_idx}/{route_num}")
        total_len = len(dataset)
        # 按照全局索引模 route_num 的方式平均切分
        indices = [i for i in range(total_len) if (i % route_num) == route_idx]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Route {route_idx + 1}/{route_num}: use {len(indices)} samples out of {total_len}.")
    
    return dataset