"""数据集参数配置 - 配置驱动方式"""

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DatasetConfig:
    """数据集配置类"""
    name: str
    display_name: str
    category: str  # 一级分类：audio, video, omni
    paths: Dict[str, str]
    subcategory: str = ""  # 二级分类：asr, qa, understanding等
    default_enabled: bool = False
    description: str = ""


DATASET_REGISTRY = [
    # 音频数据集
    DatasetConfig(
        name="gigaspeech_test", 
        display_name="GigaSpeech Test", 
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/gigaspeech/test_files/",
            "annotation_path": "./data/audio/asr/gigaspeech/test.jsonl"
        }
    ),
    DatasetConfig(
        name="wenetspeech_test_net", 
        display_name="WenetSpeech Test Net", 
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/wenetspeech/test_net/",
            "annotation_path": "./data/audio/asr/wenetspeech/test_net.jsonl"
        },
        description="WenetSpeech test_net split (24,774 samples)"
    ),
    DatasetConfig(
        name="wenetspeech_test_meeting", 
        display_name="WenetSpeech Test Meeting", 
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/wenetspeech/test_meeting/",
            "annotation_path": "./data/audio/asr/wenetspeech/test_meeting.jsonl"
        },
        description="WenetSpeech test_meeting split (8,370 samples)"
    ),
    
    # 新增的ASR数据集
    DatasetConfig(
        name="librispeech_test_clean",
        display_name="LibriSpeech Test Clean",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/",
            "annotation_path": "./data/audio/asr/librispeech/test_clean.jsonl"
        },
        description="LibriSpeech test-clean split (5.4 hours, 2620 samples)"
    ),
    DatasetConfig(
        name="librispeech_test_other",
        display_name="LibriSpeech Test Other",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/",
            "annotation_path": "./data/audio/asr/librispeech/test_other.jsonl"
        },
        description="LibriSpeech test-other split (5.3 hours, 2939 samples)"
    ),
    DatasetConfig(
        name="librispeech_dev_clean",
        display_name="LibriSpeech Dev Clean",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/",
            "annotation_path": "./data/audio/asr/librispeech/dev_clean.jsonl"
        },
        description="LibriSpeech dev-clean split for validation"
    ),
    DatasetConfig(
        name="librispeech_dev_other",
        display_name="LibriSpeech Dev Other",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/",
            "annotation_path": "./data/audio/asr/librispeech/dev_other.jsonl"
        },
        description="LibriSpeech dev-other split for validation"
    ),
    DatasetConfig(
        name="commonvoice_en",
        display_name="CommonVoice English v15",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/commonvoice/audios/",
            "annotation_path": "./data/audio/asr/commonvoice/en_v15_test.jsonl"
        },
        description="CommonVoice English v15 test set (16386 samples)"
    ),
    DatasetConfig(
        name="commonvoice_zh",
        display_name="CommonVoice Chinese v15",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/commonvoice/audios/",
            "annotation_path": "./data/audio/asr/commonvoice/zh_v15_test.jsonl"
        },
        description="CommonVoice Chinese v15 test set (10625 samples)"
    ),
    DatasetConfig(
        name="commonvoice_yue",
        display_name="CommonVoice Cantonese v15",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/commonvoice/audios/",
            "annotation_path": "./data/audio/asr/commonvoice/yue_v15_test.jsonl"
        },
        description="CommonVoice Cantonese v15 test set (5593 samples)"
    ),
    DatasetConfig(
        name="commonvoice_fr",
        display_name="CommonVoice French v15",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/commonvoice/audios/",
            "annotation_path": "./data/audio/asr/commonvoice/fr_v15_test.jsonl"
        },
        description="CommonVoice French v15 test set (16132 samples)"
    ),
    DatasetConfig(
        name="aishell1_test",
        display_name="AISHELL-1 Test",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/aishell1/",
            "annotation_path": "./data/audio/asr/aishell1/test.jsonl"
        },
        description="AISHELL-1 Chinese ASR test set (7176 samples, 约10小时)"
    ),
    DatasetConfig(
        name="aishell2_test",
        display_name="AISHELL-2 Test",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/aishell2/",
            "annotation_path": "./data/audio/asr/aishell2/test.jsonl"
        },
        description="AISHELL-2 Chinese ASR test set (5000 samples, 约10小时)"
    ),
    DatasetConfig(
        name="kespeech_test",
        display_name="KeSpeech Test",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/kespeech/",
            "annotation_path": "./data/audio/asr/kespeech/test/output.jsonl"
        },
        description="KeSpeech Chinese multi-dialect ASR test set (16768 samples)"
    ),
    DatasetConfig(
        name="voxpopuli_en",
        display_name="VoxPopuli English",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/voxpopuli/data/en/test/test_part_0/",
            "annotation_path": "./data/audio/asr/voxpopuli/data/en/asr_test.jsonl"
        },
        description="VoxPopuli English parliament speeches (1842 samples)"
    ),


    # VoiceBench QA数据集
    DatasetConfig(
        name="voicebench_alpacaeval",
        display_name="VoiceBench AlpacaEval",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/alpacaeval/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench AlpacaEval QA dataset (199 samples)"
    ),
    DatasetConfig(
        name="voicebench_alpacaeval_full",
        display_name="VoiceBench AlpacaEval Full",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/alpacaeval_full/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench AlpacaEval Full QA dataset (636 samples)"
    ),
    DatasetConfig(
        name="voicebench_bbh",
        display_name="VoiceBench BBH",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/bbh/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench Big-Bench Hard QA dataset (1000 samples)"
    ),
    DatasetConfig(
        name="voicebench_mmsu",
        display_name="VoiceBench MMSU",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/mmsu/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench MMSU QA dataset (3074 samples)"
    ),
    DatasetConfig(
        name="voicebench_openbookqa",
        display_name="VoiceBench OpenBookQA",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/openbookqa/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench OpenBookQA dataset (455 samples)"
    ),
    DatasetConfig(
        name="voicebench_advbench",
        display_name="VoiceBench AdvBench",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/advbench/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench AdvBench adversarial QA dataset (520 samples)"
    ),
    DatasetConfig(
        name="voicebench_commoneval",
        display_name="VoiceBench CommonEval",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/commoneval/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench CommonEval commonsense reasoning dataset (200 samples)"
    ),
    DatasetConfig(
        name="voicebench_ifeval",
        display_name="VoiceBench IFEval",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/ifeval/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench IFEval instruction following dataset (345 samples)"
    ),
    DatasetConfig(
        name="voicebench_sdqa",
        display_name="VoiceBench SD-QA",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/sd-qa/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench Spoken Dialogue QA dataset (6083 samples)"
    ),
    DatasetConfig(
        name="voice_cmmlu",
        display_name="Voice CMMLU",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voice_cmmlu/",
            "annotation_path": "./data/audio/qa/voice_cmmlu/all-google-tts.jsonl"
        },
        subcategory="qa",
        description="Voice CMMLU Chinese multi-task language understanding dataset (11582 samples)"
    ),
    DatasetConfig(
        name="voicebench_wildvoice",
        display_name="VoiceBench WildVoice",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/qa/voicebench/",
            "annotation_path": "./data/audio/qa/voicebench/wildvoice/test.jsonl"
        },
        subcategory="qa",
        description="VoiceBench WildVoice real-world QA dataset (1000 samples)"
    ),
    
    # MMAU 多任务音频理解数据集
    DatasetConfig(
        name="mmau_test_mini",
        display_name="MMAU Test Mini",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/multitask/MMAU/",
            "annotation_path": "./data/audio/multitask/MMAU/test/test.jsonl"
        },
        subcategory="qa",
        description="MMAU (Massive Multi-task Audio Understanding) test-mini dataset (985 samples)"
    ),
    
    # MMSU 多任务音频理解数据集
    DatasetConfig(
        name="mmsu_bench",
        display_name="MMSU Bench",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/multitask/MMSU/",
            "annotation_path": "./data/audio/multitask/MMSU/test/test_lines_4996.jsonl"
        },
        subcategory="qa",
        description="MMSU (Massive Multitask Speech Understanding) benchmark for comprehensive audio understanding evaluation"
    ),

    # MMAR 多任务音频理解数据集
    DatasetConfig(
        name="mmar_bench",
        display_name="MMAR Bench",
        category="audio",
        paths={
            "data_prefix_dir": "./data/audio/multitask/MMAR/",
            "annotation_path": "./data/audio/multitask/MMAR/test/test.jsonl"
        },
        subcategory="qa",
        description="MMAR (Multimodal Audio Reasoning) benchmark for audio reasoning and understanding tasks"
    ),

    # Audio Web Questions 和 Audio Trivia QA 数据集
    DatasetConfig(
        name="audio_web_questions",
        display_name="Audio Web Questions",
        category="audio",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/audio/qa",
            "annotation_path": "./data/audio/qa/audio_web_questions/test_audio.jsonl"
        },
        description="Audio Web Questions dataset with spoken questions and multi-answer support (2032 samples)"
    ),
    DatasetConfig(
        name="audio_trivia_qa",
        display_name="Audio Trivia QA",
        category="audio",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/audio/qa",
            "annotation_path": "./data/audio/qa/audio_trivia_qa/test-1024-clean.jsonl"
        },
        description="Audio Trivia QA dataset with spoken trivia questions and multi-answer support (1024 samples)"
    ),
    
    DatasetConfig(
        name="fleurs_zh",
        display_name="FLEURS 中文",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/fleurs/zh/",
            "annotation_path": "./data/audio/asr/fleurs/zh/test.jsonl"
        },
        description="FLEURS 中文测试集 (945 samples)"
    ),
    DatasetConfig(
        name="fleurs_en",
        display_name="FLEURS 英文",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/fleurs/en_us/",
            "annotation_path": "./data/audio/asr/fleurs/en_us/test.jsonl"
        },
        description="FLEURS 英文测试集 (647 samples)"
    ),
    DatasetConfig(
        name="peoples_speech_test",
        display_name="People's Speech Test",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/peoples_speech/test/test/",
            "annotation_path": "./data/audio/asr/peoples_speech/test.jsonl"
        },
        description="People's Speech large-scale English ASR test set (34,898 samples)"
    ),
    DatasetConfig(
        name="tedlium3_test",
        display_name="TED-LIUM v3 Test",
        category="audio",
        subcategory="asr",
        paths={
            "data_prefix_dir": "./data/audio/asr/tedlium/",
            "annotation_path": "./data/audio/asr/tedlium/TEDLIUM_release3_test.jsonl"
        },
        description="TED-LIUM v3 English speech test set (3,737 samples)"
    ),
    
    # Audio Caption datasets
    DatasetConfig(
        name="audiocaps_test",
        display_name="AudioCaps Test",
        category="audio",
        subcategory="caption",
        paths={
            "data_prefix_dir": "./data/audio/caption/AudioCaps/",
            "annotation_path": "./data/audio/caption/AudioCaps/test.jsonl"
        },
        description="AudioCaps test set for audio captioning (3,986 samples)"
    ),
    DatasetConfig(
        name="clothocaption_test",
        display_name="ClothoCaption Test",
        category="audio",
        subcategory="caption",
        paths={
            "data_prefix_dir": "./data/audio/caption/ClothoCaption/evaluation/",
            "annotation_path": "./data/audio/caption/ClothoCaption/test.jsonl"
        },
        description="ClothoCaption test set for audio captioning (1,046 samples)"
    ),
    DatasetConfig(
        name="wavcaps_audioset_sl",
        display_name="WavCaps AudioSet_SL",
        category="audio",
        subcategory="caption",
        paths={
            "data_prefix_dir": "./data/audio/caption/WavCaps/",
            "annotation_path": "./data/audio/caption/WavCaps/AudioSet_SL_test.jsonl"
        },
        description="WavCaps AudioSet_SL test set for audio captioning (11,677 samples)"
    ),
    DatasetConfig(
        name="wavcaps_freesound",
        display_name="WavCaps FreeSound",
        category="audio",
        subcategory="caption",
        paths={
            "data_prefix_dir": "./data/audio/caption/WavCaps/",
            "annotation_path": "./data/audio/caption/WavCaps/FreeSound_test.jsonl"
        },
        description="WavCaps FreeSound test set for audio captioning (1,061 samples)"
    ),
    DatasetConfig(
        name="wavcaps_soundbible",
        display_name="WavCaps SoundBible",
        category="audio",
        subcategory="caption",
        paths={
            "data_prefix_dir": "./data/audio/caption/WavCaps/",
            "annotation_path": "./data/audio/caption/WavCaps/SoundBible_test.jsonl"
        },
        description="WavCaps SoundBible test set for audio captioning (1,233 samples)"
    ),
    
    # Audio Speech Translation (AST) datasets
    DatasetConfig(
        name="covost2_zh_en",
        display_name="CoVoST2 ZH-EN",
        category="audio",
        subcategory="ast",
        paths={
            "data_prefix_dir": "./data/audio/ast/covost2-zh-en/audio/",
            "annotation_path": "./data/audio/ast/covost2-zh-en/covost_v2.zh-CN_en_test.jsonl"
        },
        description="CoVoST2 Chinese to English speech translation test set (4,897 samples)"
    ),
    DatasetConfig(
        name="covost2_en_zh",
        display_name="CoVoST2 EN-ZH",
        category="audio",
        subcategory="ast",
        paths={
            "data_prefix_dir": "./data/audio/ast/covost2-en-zh/audio/",
            "annotation_path": "./data/audio/ast/covost2-en-zh/covost_v2.en_zh-CN_test.jsonl"
        },
        description="CoVoST2 English to Chinese speech translation test set (15,530 samples)"
    ),
    
    # Audio Classification datasets
    DatasetConfig(
        name="vocalsound",
        display_name="VocalSound",
        category="audio",
        subcategory="cls",
        paths={
            "data_prefix_dir": "./data/audio/cls/vocalsound/",
            "annotation_path": "./data/audio/cls/vocalsound/test.jsonl"
        },
        description="VocalSound dataset for human vocal sound classification (3,592 samples)"
    ),
    
    # MELD 情感识别数据集
    DatasetConfig(
        name="meld",
        display_name="MELD",
        category="audio",
        subcategory="cls",
        paths={
            "data_prefix_dir": "./data/audio/cls/MELD/",
            "annotation_path": "./data/audio/cls/MELD/test_sent_emo_audio.jsonl"
        },
        description="MELD (Multimodal EmotionLines Dataset) for emotion recognition in conversations (2,610 samples)"
    ),

    # LiveSports-3K CC (统一格式，用于双工推理)
    # 使用过滤后的文件，排除了 116 个视频缺失的事件
    DatasetConfig(
        name="livesports3k_cc",
        display_name="LiveSports-3K CC",
        category="duplex_video",
        subcategory="caption",
        paths={
            "annotation_path": "./data/omni/livecc-bench/livecc_bench_cc.jsonl",
            "data_prefix_dir": "./data/omni/livesports3k/"
        },
        description="LiveSports-3K CC: 1586 video caption samples for duplex streaming inference (filtered, excluding 116 missing videos)"
    ),
    
    # Daily-Omni 数据集
    DatasetConfig(
        name="daily_omni",
        display_name="Daily-Omni",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/daily-omni/",
            "annotation_path": "./data/omni/raw_hf/daily-omni/daily_omni.jsonl"
        },
        description="Daily-Omni benchmark: 1197 audio-visual QA samples across 6 types"
    ),
    
    # OmniBench 数据集
    DatasetConfig(
        name="omnibench",
        display_name="OmniBench",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/omnibench/",
            "annotation_path": "./data/omni/raw_hf/omnibench/omnibench.jsonl"
        },
        description="OmniBench: 1142 image+audio QA samples across 8 task types"
    ),
    
    # Video-MME 数据集（统一 Omni JSONL）
    DatasetConfig(
        name="videomme",
        display_name="Video-MME",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/videomme/",
            "annotation_path": "./data/omni/raw_hf/videomme/videomme.jsonl",
        },
        description="Video-MME: 2700 video QA samples with audio (converted to unified omni JSONL format)"
    ),
    # Video-MME Short 子集（duration=short，仅短视频）
    DatasetConfig(
        name="videomme_short",
        display_name="Video-MME (Short)",
        category="omni",
        subcategory="qa",
        paths={
            # 使用离线生成的 short 子集 JSONL
            "data_prefix_dir": "./data/omni/raw_hf/videomme/",
            "annotation_path": "./data/omni/raw_hf/videomme/videomme_short.jsonl",
        },
        description="Video-MME short-duration subset (duration=short) stored in a separate omni JSONL"
    ),
    
    # UNO-Bench 数据集（完整版：MCQ + Open-Ended）
    DatasetConfig(
        name="unobench",
        display_name="UNO-Bench",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/uno-bench/",
            "annotation_path": "./data/omni/raw_hf/uno-bench/unobench.jsonl"
        },
        description="UNO-Bench: 3730 unified omni-modal samples (audio/image/video) for compositional law evaluation"
    ),
    
    # UNO-Bench MCQ 部分（只包含多选题）
    DatasetConfig(
        name="unobench_mc",
        display_name="UNO-Bench-MC",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/uno-bench/",
            "annotation_path": "./data/omni/raw_hf/uno-bench/unobench_mc.jsonl"
        },
        description="UNO-Bench MCQ only: 3274 multi-choice questions (audio/image/video)"
    ),
    
    # WorldSense 数据集
    DatasetConfig(
        name="worldsense",
        display_name="WorldSense",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/worldsense/",
            "annotation_path": "./data/omni/raw_hf/worldsense/worldsense.jsonl"
        },
        description="WorldSense: 3172 real-world video QA samples across 26 task types"
    ),
    
    # AV-Odyssey 数据集
    DatasetConfig(
        name="av_odyssey",
        display_name="AV-Odyssey",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/av-odyssey/",
            "annotation_path": "./data/omni/raw_hf/av-odyssey/av_odyssey.jsonl"
        },
        description="AV-Odyssey: 4555 audio-visual MCQ samples across 26 task types (7 modality combinations)"
    ),
    
    # Video-Holmes 数据集
    DatasetConfig(
        name="video_holmes",
        display_name="Video-Holmes",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/video-holmes/",
            "annotation_path": "./data/omni/raw_hf/video-holmes/video_holmes.jsonl"
        },
        description="Video-Holmes: 1837 MCQ samples for complex video reasoning across 7 task types (SR, IMC, TCI, TA, MHR, CTI, PAR)"
    ),
    
    # AVUT-Benchmark Human 数据集
    DatasetConfig(
        name="avut_benchmark_human",
        display_name="AVUT-Benchmark Human",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/avut-benchmark/",
            "annotation_path": "./data/omni/raw_hf/avut-benchmark/annotation/avut_benchmark_human.jsonl"
        },
        description="AVUT-Benchmark Human: 1734 human-annotated audio-visual MCQ samples across 6 task types and 14 video types"
    ),
    
    # AVUT-Benchmark Gemini 数据集
    DatasetConfig(
        name="avut_benchmark_gemini",
        display_name="AVUT-Benchmark Gemini",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/avut-benchmark/",
            "annotation_path": "./data/omni/raw_hf/avut-benchmark/annotation/avut_benchmark_gemini.jsonl"
        },
        description="AVUT-Benchmark Gemini: 9874 Gemini-generated audio-visual MCQ samples across 6 task types and 10 video types"
    ),
    
    # OVO-Bench（Omni 统一格式：离线 MCQ/QA）
    DatasetConfig(
        name="ovobench",
        display_name="OVO-Bench",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/ovobench/",
            "annotation_path": "./data/omni/raw_hf/ovobench/ovobench.jsonl"
        },
        description="OVOBench converted to unified omni JSONL format for offline video QA/MCQ evaluation"
    ),
    
    # StreamingBench-Real（统一 Omni JSONL：离线 MCQ）
    DatasetConfig(
        name="streamingbench_real",
        display_name="StreamingBench-Real (Offline)",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/streamingbench/video/",
            "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_real.jsonl"
        },
        description="StreamingBench Real-Time Visual Understanding subset converted to unified omni JSONL format (offline, video-only)"
    ),
    # StreamingBench-Omni-Fix（筛选后的 Omni JSONL：仅包含 Emotion Recognition, Scene Understanding, Multimodal Alignment, Source Discrimination）
    DatasetConfig(
        name="streamingbench_omni_fix",
        display_name="StreamingBench-Omni-Fix (Offline)",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/streamingbench/video/",
            "annotation_path": "./data/omni/raw_hf/streamingbench/streamingbench_omni_fix.jsonl"
        },
        description="StreamingBench omni-fix subset: only Emotion Recognition, Scene Understanding, Multimodal Alignment, Source Discrimination (1000 samples)"
    ),
    # StreamingBench-SQA（统一 Omni JSONL：离线 MCQ）
    DatasetConfig(
        name="streamingbench_sqa",
        display_name="StreamingBench-SQA (Offline)",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/streamingbench/video/",
            "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_sqa.jsonl"
        },
        description="StreamingBench Sequential Question Answering subset converted to unified omni JSONL format (offline, with textual context)"
    ),

    # FutureOmni 未来预测数据集
    DatasetConfig(
        name="futureomni",
        display_name="FutureOmni",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/futureomni/",
            "annotation_path": "./data/omni/raw_hf/futureomni/annotation/futureomni.jsonl"
        },
        description="FutureOmni: 1034 MCQ samples for evaluating future forecasting from omni-modal (audio-visual) context"
    ),
    
    # 训练数据集
    # JointAVBench（音视频联合理解 MCQ）
    DatasetConfig(
        name="jointavbench",
        display_name="JointAVBench",
        category="omni",
        subcategory="qa",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/JointAVBench/",
            "annotation_path": "./data/omni/raw_hf/JointAVBench/converted/jointavbench.jsonl"
        },
        description="JointAVBench: Joint Audio-Video Benchmark for multi-modal understanding (MCQ, 15 task types)"
    ),
    
    # OV-AVEL: Open-Vocabulary Audio-Visual Event Localization
    # 参考论文: https://arxiv.org/pdf/2411.11278
    DatasetConfig(
        name="ovavel",
        display_name="OV-AVEL Test",
        category="omni_duplex",
        subcategory="event_localization",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/ovavel/merged/",
            "annotation_path": "./data/omni/raw_hf/ovavel/annotation/ovavel_test.jsonl"
        },
        description="OV-AVEBench Test: 完整测试集 (5820 samples, base: 1664, novel: 4156)"
    ),
    
    # AVMeme-Exam: 音视频 Meme 理解评测基准
    # 参考论文: https://arxiv.org/pdf/2601.17645
    # visual_cheat=true 的样本只有 WavPath，visual_cheat=false 的样本有 VideoPath
    DatasetConfig(
        name="avmeme_full",
        display_name="AVMeme-Exam Full",
        category="omni",
        subcategory="meme_understanding",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/avmeme/",
            "annotation_path": "./data/omni/raw_hf/avmeme/annotation/avmeme_full.jsonl"
        },
        description="AVMeme-Exam Full: 完整数据集 (1032 samples, 131 visual_cheat)"
    ),
    DatasetConfig(
        name="avmeme_main",
        display_name="AVMeme-Exam Main",
        category="omni",
        subcategory="meme_understanding",
        paths={
            "data_prefix_dir": "./data/omni/raw_hf/avmeme/",
            "annotation_path": "./data/omni/raw_hf/avmeme/annotation/avmeme_main.jsonl"
        },
        description="AVMeme-Exam Main: 去除 text_cheat 后的数据集 (846 samples, 118 visual_cheat)"
    ),
]

# 额外的静态配置
STATIC_CONFIGS: Dict[str, Any] = {
    "ovobench_tasks": ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
    "streaming_context_time": 60,
    "streamingbench_tasks": ["real", "sqa"],
    
    # Duplex相关配置
    "duplex_img_size": 448,
    "duplex_img_aug": False,
    "duplex_use_adaptive_slice": False,
    "duplex_total_max_length": 8192,
    "duplex_insert_listen_speak": False,
    "duplex_audio_pool_step": 5,
    "duplex_audio_chunk_length": 0.1,
    "duplex_streaming_text_reserved_len": 300,
    "duplex_insert_vad": False,
    "duplex_audio_normalize": False,
    "duplex_stable_audio_token_interval": True,
}


def add_dataset_args(parser: argparse.ArgumentParser):
    """动态添加数据集参数"""
    dataset_group = parser.add_argument_group('数据集配置')
    
    for key, default_value in STATIC_CONFIGS.items():
        if key == "ovobench_tasks":
            dataset_group.add_argument(f"--ovobench_task", type=str, nargs="+", 
                                     choices=default_value, default=default_value)
        elif key == "streamingbench_tasks":
            dataset_group.add_argument(f"--streamingbench_tasks", type=str, nargs="+", 
                                     choices=default_value, default=default_value,
                                     help="StreamingBench任务列表：real(实时), omni(多模态), sqa(序列问答), proactive(主动输出)")
        elif key == "streaming_context_time":
            dataset_group.add_argument(f"--{key}", type=int, default=default_value,
                                     help="StreamingBench上下文时间长度（秒）")
        # 处理duplex相关的数值型参数
        elif key in ["duplex_img_size", "duplex_total_max_length", "duplex_audio_pool_step", 
                     "duplex_streaming_text_reserved_len"]:
            dataset_group.add_argument(f"--{key}", type=int, default=default_value)
        elif key in ["duplex_audio_chunk_length"]:
            dataset_group.add_argument(f"--{key}", type=float, default=default_value)
        elif key in ["duplex_img_aug", "duplex_use_adaptive_slice", "duplex_insert_listen_speak",
                     "duplex_insert_vad", "duplex_audio_normalize", "duplex_stable_audio_token_interval"]:
            dataset_group.add_argument(f"--{key}", action="store_true", default=default_value)
        else:
            dataset_group.add_argument(f"--{key}", type=str, default=default_value)
    
    for config in DATASET_REGISTRY:
        for path_key, default_path in config.paths.items():
            arg_name = f"--{config.name}_{path_key}"
            dataset_group.add_argument(arg_name, type=str, default=default_path)
    
    # 添加LiveCC特定的参数
    dataset_group.add_argument("--livecc_data_type", type=str, default="clipped", 
                              choices=["clipped", "frames"],
                              help="LiveCC数据类型：clipped (默认使用test_cc_clipped.jsonl) 或 frames (使用test_cc_frames_1fps.jsonl)")


def add_evaluation_flags(parser: argparse.ArgumentParser):
    """动态添加评估标志"""
    eval_group = parser.add_argument_group('评估控制')
    
    for config in DATASET_REGISTRY:
        eval_group.add_argument(f"--eval_{config.name}", action="store_true", 
                               default=config.default_enabled)
    
    # 手动添加StreamingBench评估标志
    eval_group.add_argument("--eval_streamingbench", action="store_true", default=False,
                           help="启用StreamingBench评估")
    
    categories = set(config.category for config in DATASET_REGISTRY)
    for category in categories:
        eval_group.add_argument(f"--eval_all_{category}", action="store_true", default=False)
    
    eval_group.add_argument("--eval_all", action="store_true", default=False)


def apply_evaluation_logic(args):
    """应用评估逻辑：处理eval_all和eval_all_category标志"""
    
    if args.eval_all:
        for config in DATASET_REGISTRY:
            setattr(args, f"eval_{config.name}", True)
        return
    
    categories = set(config.category for config in DATASET_REGISTRY)
    for category in categories:
        category_flag = f"eval_all_{category}"
        if hasattr(args, category_flag) and getattr(args, category_flag):
            for config in DATASET_REGISTRY:
                if config.category == category:
                    setattr(args, f"eval_{config.name}", True)


def get_dataset_info():
    """获取数据集信息摘要"""
    by_category = {}
    by_subcategory = {}
    for config in DATASET_REGISTRY:
        if config.category not in by_category:
            by_category[config.category] = []
        by_category[config.category].append(config)
        
        # 统计subcategory
        if config.subcategory:
            key = f"{config.category}/{config.subcategory}"
            if key not in by_subcategory:
                by_subcategory[key] = []
            by_subcategory[key].append(config)
    
    return {
        'total': len(DATASET_REGISTRY),
        'by_category': by_category,
        'by_subcategory': by_subcategory,
        'categories': list(by_category.keys())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配置驱动的数据集参数")
    add_dataset_args(parser)
    add_evaluation_flags(parser)
    
    print("🧪 配置驱动的数据集参数系统")
    print("=" * 50)
    
    info = get_dataset_info()
    print(f"📊 注册的数据集 (总计: {info['total']} 个):")
    
    # 按category和subcategory分组显示
    for category, configs in info['by_category'].items():
        print(f"\n  📁 {category.upper()} ({len(configs)} datasets):")
        # 按subcategory分组
        subcategories = {}
        no_subcategory = []
        for config in configs:
            if config.subcategory:
                if config.subcategory not in subcategories:
                    subcategories[config.subcategory] = []
                subcategories[config.subcategory].append(config)
            else:
                no_subcategory.append(config)
        
        # 显示有subcategory的数据集
        for subcat, subconfigs in sorted(subcategories.items()):
            print(f"    └─ {subcat}: {[c.display_name for c in subconfigs]}")
        
        # 显示没有subcategory的数据集
        if no_subcategory:
            print(f"    └─ other: {[c.display_name for c in no_subcategory]}")
    
    print(f"\n🚀 支持的评估命令示例:")
    print(f"  --eval_all_audio     # 启用所有音频数据集")
    print(f"  --eval_all_video     # 启用所有视频数据集")
    print(f"  --eval_all           # 启用所有数据集")
    print(f"  --eval_gigaspeech_test --eval_aishell1_test  # 启用特定数据集")
    
    print(f"\n✅ 动态生成了 {len(DATASET_REGISTRY)} 个数据集的参数和评估标志")