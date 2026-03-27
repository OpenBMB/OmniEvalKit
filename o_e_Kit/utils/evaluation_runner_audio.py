"""
音频数据集评估模块
显式处理每个音频数据集的评估

支持异步评估模式：通过 async_evaluate 参数控制
"""

import torch
from o_e_Kit.utils.infer import infer_and_evaluate
from o_e_Kit.utils.dataset_loader import load_dataset


def evaluate_all_audio_datasets(args, model, time, async_evaluate: bool = False):
    """评估所有音频数据集（ASR、QA和Caption）- 显式处理每个数据集
    
    Args:
        args: 命令行参数
        model: 模型实例
        time: 时间戳
        async_evaluate: 是否使用异步评估模式
    
    Returns:
        评估结果字典
    """
    result = {}
    
    # ================== 中文ASR数据集 ==================
    if args.max_sample_num == -1:
        args.max_sample_num = None
        
    # WenetSpeech Net评估
    if args.eval_wenetspeech_test_net:
        print("\nEvaluating WenetSpeech Test Net...")
        dataset = load_dataset(args, "wenetspeech_test_net")
        result['wenetspeech_test_net'] = infer_and_evaluate(
            model, dataset, args.model_name, "wenetspeech_test_net", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"WenetSpeech Net result: {result['wenetspeech_test_net']}")
    
    # WenetSpeech Meeting评估
    if args.eval_wenetspeech_test_meeting:
        print("\nEvaluating WenetSpeech Test Meeting...")
        dataset = load_dataset(args, "wenetspeech_test_meeting")
        result['wenetspeech_test_meeting'] = infer_and_evaluate(
            model, dataset, args.model_name, "wenetspeech_test_meeting", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"WenetSpeech Meeting result: {result['wenetspeech_test_meeting']}")
    
    # AISHELL-1评估
    if args.eval_aishell1_test:
        print("\nEvaluating AISHELL-1 Test...")
        dataset = load_dataset(args, "aishell1_test")
        result['aishell1_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "aishell1_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"AISHELL-1 result: {result['aishell1_test']}")
    
    # AISHELL-2评估
    if args.eval_aishell2_test:
        print("\nEvaluating AISHELL-2 Test...")
        dataset = load_dataset(args, "aishell2_test")
        result['aishell2_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "aishell2_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"AISHELL-2 result: {result['aishell2_test']}")
    
    # KeSpeech评估
    if getattr(args, 'eval_kespeech_test', False):
        print("\nEvaluating KeSpeech Test...")
        dataset = load_dataset(args, "kespeech_test")
        result['kespeech_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "kespeech_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"KeSpeech result: {result['kespeech_test']}")
    
    # CommonVoice Chinese评估
    if args.eval_commonvoice_zh:
        print("\nEvaluating CommonVoice Chinese...")
        dataset = load_dataset(args, "commonvoice_zh")
        result['commonvoice_zh'] = infer_and_evaluate(
            model, dataset, args.model_name, "commonvoice_zh", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"CommonVoice Chinese result: {result['commonvoice_zh']}")
    
    # FLEURS Chinese评估
    if args.eval_fleurs_zh:
        print("\nEvaluating FLEURS Chinese...")
        dataset = load_dataset(args, "fleurs_zh")
        result['fleurs_zh'] = infer_and_evaluate(
            model, dataset, args.model_name, "fleurs_zh", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"FLEURS Chinese result: {result['fleurs_zh']}")
    
    # ================== 英文ASR数据集 ==================
    
    # GigaSpeech评估
    if args.eval_gigaspeech_test:
        print("\nEvaluating GigaSpeech Test...")
        dataset = load_dataset(args, "gigaspeech_test")
        result['gigaspeech_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "gigaspeech_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"GigaSpeech result: {result['gigaspeech_test']}")
    
    # LibriSpeech Test Clean评估
    if args.eval_librispeech_test_clean:
        print("\nEvaluating LibriSpeech Test Clean...")
        dataset = load_dataset(args, "librispeech_test_clean")
        result['librispeech_test_clean'] = infer_and_evaluate(
            model, dataset, args.model_name, "librispeech_test_clean", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"LibriSpeech Test Clean result: {result['librispeech_test_clean']}")
    
    # LibriSpeech Test Other评估
    if args.eval_librispeech_test_other:
        print("\nEvaluating LibriSpeech Test Other...")
        dataset = load_dataset(args, "librispeech_test_other")
        result['librispeech_test_other'] = infer_and_evaluate(
            model, dataset, args.model_name, "librispeech_test_other", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"LibriSpeech Test Other result: {result['librispeech_test_other']}")
    
    # LibriSpeech Dev Clean评估
    if args.eval_librispeech_dev_clean:
        print("\nEvaluating LibriSpeech Dev Clean...")
        dataset = load_dataset(args, "librispeech_dev_clean")
        result['librispeech_dev_clean'] = infer_and_evaluate(
            model, dataset, args.model_name, "librispeech_dev_clean", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"LibriSpeech Dev Clean result: {result['librispeech_dev_clean']}")
    
    # LibriSpeech Dev Other评估
    if args.eval_librispeech_dev_other:
        print("\nEvaluating LibriSpeech Dev Other...")
        dataset = load_dataset(args, "librispeech_dev_other")
        result['librispeech_dev_other'] = infer_and_evaluate(
            model, dataset, args.model_name, "librispeech_dev_other", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"LibriSpeech Dev Other result: {result['librispeech_dev_other']}")
    
    # CommonVoice English评估
    if args.eval_commonvoice_en:
        print("\nEvaluating CommonVoice English...")
        dataset = load_dataset(args, "commonvoice_en")
        result['commonvoice_en'] = infer_and_evaluate(
            model, dataset, args.model_name, "commonvoice_en", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"CommonVoice English result: {result['commonvoice_en']}")
    
    # VoxPopuli English评估
    if args.eval_voxpopuli_en:
        print("\nEvaluating VoxPopuli English...")
        dataset = load_dataset(args, "voxpopuli_en")
        result['voxpopuli_en'] = infer_and_evaluate(
            model, dataset, args.model_name, "voxpopuli_en", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoxPopuli English result: {result['voxpopuli_en']}")
    
    # FLEURS English评估
    if args.eval_fleurs_en:
        print("\nEvaluating FLEURS English...")
        dataset = load_dataset(args, "fleurs_en")
        result['fleurs_en'] = infer_and_evaluate(
            model, dataset, args.model_name, "fleurs_en", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"FLEURS English result: {result['fleurs_en']}")
    
    # People's Speech评估
    if args.eval_peoples_speech_test:
        print("\nEvaluating People's Speech Test...")
        dataset = load_dataset(args, "peoples_speech_test")
        result['peoples_speech_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "peoples_speech_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"People's Speech result: {result['peoples_speech_test']}")
    
    # TED-LIUM v3评估
    if args.eval_tedlium3_test:
        print("\nEvaluating TED-LIUM v3 Test...")
        dataset = load_dataset(args, "tedlium3_test")
        result['tedlium3_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "tedlium3_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"TED-LIUM v3 result: {result['tedlium3_test']}")
    
    # ================== 其他语言ASR数据集 ==================
    
    # CommonVoice Cantonese评估
    if args.eval_commonvoice_yue:
        print("\nEvaluating CommonVoice Cantonese...")
        dataset = load_dataset(args, "commonvoice_yue")
        result['commonvoice_yue'] = infer_and_evaluate(
            model, dataset, args.model_name, "commonvoice_yue", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"CommonVoice Cantonese result: {result['commonvoice_yue']}")
    
    # CommonVoice French评估
    if args.eval_commonvoice_fr:
        print("\nEvaluating CommonVoice French...")
        dataset = load_dataset(args, "commonvoice_fr")
        result['commonvoice_fr'] = infer_and_evaluate(
            model, dataset, args.model_name, "commonvoice_fr", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"CommonVoice French result: {result['commonvoice_fr']}")
    
    # ================== QA数据集评估 ==================
    
    # VoiceBench系列数据集评估
    evaluate_voicebench_datasets(args, model, time, result, async_evaluate=async_evaluate)
    
    # Audio Caption系列数据集评估
    evaluate_caption_datasets(args, model, time, result, async_evaluate=async_evaluate)
    
    # Audio Classification系列数据集评估
    evaluate_classification_datasets(args, model, time, result, async_evaluate=async_evaluate)
    
    return result


def evaluate_voicebench_datasets(args, model, time, result, async_evaluate: bool = False):
    """评估所有VoiceBench数据集
    
    Args:
        async_evaluate: 是否使用异步评估模式
    """
    
    # VoiceBench AlpacaEval评估
    if args.eval_voicebench_alpacaeval:
        print("\nEvaluating VoiceBench AlpacaEval...")
        dataset = load_dataset(args, "voicebench_alpacaeval")
        result['voicebench_alpacaeval'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_alpacaeval", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench AlpacaEval result: {result['voicebench_alpacaeval']}")
    
    # VoiceBench AlpacaEval Full评估
    if args.eval_voicebench_alpacaeval_full:
        print("\nEvaluating VoiceBench AlpacaEval Full...")
        dataset = load_dataset(args, "voicebench_alpacaeval_full")
        result['voicebench_alpacaeval_full'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_alpacaeval_full", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench AlpacaEval Full result: {result['voicebench_alpacaeval_full']}")
    
    # VoiceBench BBH评估
    if args.eval_voicebench_bbh:
        print("\nEvaluating VoiceBench BBH...")
        dataset = load_dataset(args, "voicebench_bbh")
        result['voicebench_bbh'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_bbh", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench BBH result: {result['voicebench_bbh']}")
    
    # VoiceBench MMSU评估
    if args.eval_voicebench_mmsu:
        print("\nEvaluating VoiceBench MMSU...")
        dataset = load_dataset(args, "voicebench_mmsu")
        result['voicebench_mmsu'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_mmsu", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench MMSU result: {result['voicebench_mmsu']}")
    
    # VoiceBench OpenBookQA评估
    if args.eval_voicebench_openbookqa:
        print("\nEvaluating VoiceBench OpenBookQA...")
        dataset = load_dataset(args, "voicebench_openbookqa")
        result['voicebench_openbookqa'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_openbookqa", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench OpenBookQA result: {result['voicebench_openbookqa']}")
    
    # VoiceBench AdvBench评估
    if args.eval_voicebench_advbench:
        print("\nEvaluating VoiceBench AdvBench...")
        dataset = load_dataset(args, "voicebench_advbench")
        result['voicebench_advbench'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_advbench", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench AdvBench result: {result['voicebench_advbench']}")
    
    # VoiceBench CommonEval评估
    if args.eval_voicebench_commoneval:
        print("\nEvaluating VoiceBench CommonEval...")
        dataset = load_dataset(args, "voicebench_commoneval")
        result['voicebench_commoneval'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_commoneval", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench CommonEval result: {result['voicebench_commoneval']}")
    
    # VoiceBench IFEval评估
    if args.eval_voicebench_ifeval:
        print("\nEvaluating VoiceBench IFEval...")
        dataset = load_dataset(args, "voicebench_ifeval")
        result['voicebench_ifeval'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_ifeval", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench IFEval result: {result['voicebench_ifeval']}")
    
    # VoiceBench SDQA评估
    if args.eval_voicebench_sdqa:
        print("\nEvaluating VoiceBench SDQA...")
        dataset = load_dataset(args, "voicebench_sdqa")
        result['voicebench_sdqa'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_sdqa", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench SDQA result: {result['voicebench_sdqa']}")
    
    # Voice CMMLU评估
    if args.eval_voice_cmmlu:
        print("\nEvaluating Voice CMMLU...")
        dataset = load_dataset(args, "voice_cmmlu")
        result['voice_cmmlu'] = infer_and_evaluate(
            model, dataset, args.model_name, "voice_cmmlu", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"Voice CMMLU result: {result['voice_cmmlu']}")
    
    # Audio Trivia QA评估
    if args.eval_audio_trivia_qa:
        print("\nEvaluating Audio Trivia QA...")
        dataset = load_dataset(args, "audio_trivia_qa")
        result['audio_trivia_qa'] = infer_and_evaluate(
            model, dataset, args.model_name, "audio_trivia_qa", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"Audio Trivia QA result: {result['audio_trivia_qa']}")
    
    # Audio Web Questions评估
    if args.eval_audio_web_questions:
        print("\nEvaluating Audio Web Questions...")
        dataset = load_dataset(args, "audio_web_questions")
        result['audio_web_questions'] = infer_and_evaluate(
            model, dataset, args.model_name, "audio_web_questions", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"Audio Web Questions result: {result['audio_web_questions']}")
    
    # VoiceBench WildVoice评估
    if args.eval_voicebench_wildvoice:
        print("\nEvaluating VoiceBench WildVoice...")
        dataset = load_dataset(args, "voicebench_wildvoice")
        result['voicebench_wildvoice'] = infer_and_evaluate(
            model, dataset, args.model_name, "voicebench_wildvoice", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VoiceBench WildVoice result: {result['voicebench_wildvoice']}")
    
    # MMAU Test Mini评估
    if args.eval_mmau_test_mini:
        print("\nEvaluating MMAU Test Mini...")
        dataset = load_dataset(args, "mmau_test_mini")
        result['mmau_test_mini'] = infer_and_evaluate(
            model, dataset, args.model_name, "mmau_test_mini", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"MMAU Test Mini result: {result['mmau_test_mini']}")
    
    # MMSU Bench评估
    if args.eval_mmsu_bench:
        print("\nEvaluating MMSU Bench...")
        dataset = load_dataset(args, "mmsu_bench")
        result['mmsu_bench'] = infer_and_evaluate(
            model, dataset, args.model_name, "mmsu_bench", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"MMSU Bench result: {result['mmsu_bench']}")
    
    # MMAR Bench评估
    if args.eval_mmar_bench:
        print("\nEvaluating MMAR Bench...")
        dataset = load_dataset(args, "mmar_bench")
        result['mmar_bench'] = infer_and_evaluate(
            model, dataset, args.model_name, "mmar_bench", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"MMAR Bench result: {result['mmar_bench']}")


def evaluate_caption_datasets(args, model, time, result, async_evaluate: bool = False):
    """评估所有Audio Caption数据集
    
    Args:
        async_evaluate: 是否使用异步评估模式
    """
    
    # AudioCaps Test评估
    if args.eval_audiocaps_test:
        print("\nEvaluating AudioCaps Test...")
        dataset = load_dataset(args, "audiocaps_test")
        result['audiocaps_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "audiocaps_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"AudioCaps Test result: {result['audiocaps_test']}")
    
    # ClothoCaption Test评估
    if args.eval_clothocaption_test:
        print("\nEvaluating ClothoCaption Test...")
        dataset = load_dataset(args, "clothocaption_test")
        result['clothocaption_test'] = infer_and_evaluate(
            model, dataset, args.model_name, "clothocaption_test", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"ClothoCaption Test result: {result['clothocaption_test']}")
    
    # WavCaps AudioSet_SL评估
    if args.eval_wavcaps_audioset_sl:
        print("\nEvaluating WavCaps AudioSet_SL...")
        dataset = load_dataset(args, "wavcaps_audioset_sl")
        result['wavcaps_audioset_sl'] = infer_and_evaluate(
            model, dataset, args.model_name, "wavcaps_audioset_sl", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"WavCaps AudioSet_SL result: {result['wavcaps_audioset_sl']}")
    
    # WavCaps FreeSound评估
    if args.eval_wavcaps_freesound:
        print("\nEvaluating WavCaps FreeSound...")
        dataset = load_dataset(args, "wavcaps_freesound")
        result['wavcaps_freesound'] = infer_and_evaluate(
            model, dataset, args.model_name, "wavcaps_freesound", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"WavCaps FreeSound result: {result['wavcaps_freesound']}")
    
    # WavCaps SoundBible评估
    if args.eval_wavcaps_soundbible:
        print("\nEvaluating WavCaps SoundBible...")
        dataset = load_dataset(args, "wavcaps_soundbible")
        result['wavcaps_soundbible'] = infer_and_evaluate(
            model, dataset, args.model_name, "wavcaps_soundbible", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"WavCaps SoundBible result: {result['wavcaps_soundbible']}")


def evaluate_classification_datasets(args, model, time, result, async_evaluate: bool = False):
    """评估所有Audio Classification数据集
    
    Args:
        async_evaluate: 是否使用异步评估模式
    """
    
    # VocalSound评估
    if args.eval_vocalsound:
        print("\nEvaluating VocalSound...")
        dataset = load_dataset(args, "vocalsound")
        result['vocalsound'] = infer_and_evaluate(
            model, dataset, args.model_name, "vocalsound", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"VocalSound result: {result['vocalsound']}")
    
    # MELD评估
    if args.eval_meld:
        print("\nEvaluating MELD...")
        dataset = load_dataset(args, "meld")
        result['meld'] = infer_and_evaluate(
            model, dataset, args.model_name, "meld", time,
            answer_path=args.answer_path, batch_size=args.batchsize, 
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"MELD result: {result['meld']}")
    
    # CoVoST2 ZH-EN 语音翻译评估（中文转英文，使用 BLEU）
    if getattr(args, 'eval_covost2_zh_en', False):
        print("\nEvaluating CoVoST2 ZH-EN...")
        dataset = load_dataset(args, "covost2_zh_en")
        result['covost2_zh_en'] = infer_and_evaluate(
            model, dataset, args.model_name, "covost2_zh_en", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"CoVoST2 ZH-EN result: {result['covost2_zh_en']}")
    
    # CoVoST2 EN-ZH 语音翻译评估（英文转中文，使用 BLEU）
    if getattr(args, 'eval_covost2_en_zh', False):
        print("\nEvaluating CoVoST2 EN-ZH...")
        dataset = load_dataset(args, "covost2_en_zh")
        result['covost2_en_zh'] = infer_and_evaluate(
            model, dataset, args.model_name, "covost2_en_zh", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
        print(f"CoVoST2 EN-ZH result: {result['covost2_en_zh']}")