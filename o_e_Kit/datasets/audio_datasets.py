"""音频数据集模块，专门处理音频数据

统一使用JSONL格式，简化数据处理流程
所有数据集都使用 WavPath 字段表示音频路径，text/sentence 字段表示转录文本
"""

import os
from typing import List, Dict, Any
import jsonlines
import numpy as np

from o_e_Kit.datasets.base_dataset import BaseDataset
from o_e_Kit.utils.utils import get_audio_duration

class AudioEvalDataset(BaseDataset):
    def __init__(self, annotation_path: str, data_prefix_dir: str = "", 
                 dataset_name: str = '', max_duration: float = None):
        self.dataset_name = dataset_name
        self.max_duration = max_duration  # None表示不限制时长
        self.duration_stats = {
            'total_count': 0,
            'valid_count': 0,
            'durations': [],  # 记录所有有效音频时长
            'distribution': {  # 时长分布统计
                '0-10s': 0,
                '10-20s': 0,
                '20-30s': 0,
                '30-60s': 0,
                '60-120s': 0,
                '>120s': 0,
            }
        }
        super().__init__(annotation_path, data_prefix_dir, modal="audio")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        data_list = []
        unused_keys = set()  # 记录未使用的字段
        
        with jsonlines.open(self.annotation_path, mode='r') as reader:
            for ann in reader:
                if len(data_list) == 0:
                    used_keys = {'WavPath', 'audio', 'text', 'sentence', 'transcription', 'label', 'normalized_text', 
                                'answer', 'answers', 'Answer', 'question', 'reference', 'id', 'caption', 'gt', 'source',
                                'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5',  # ClothoCaption
                                'file_name', 'audiocap_id', 'youtube_id', 'start_time',  # AudioCaps
                                'choices', 'instruction_id_list', 'kwargs', 'key', 'prompt', 'task', 'duration'}  # QA/IFEval/MMAU相关字段
                    unused_keys = set(ann.keys()) - used_keys
                
                # 支持单音频和多音频两种格式
                # MMAU-Pro 使用 audio_path_list，其他数据集使用 WavPath
                if 'audio_path_list' in ann:
                    # 多音频格式 (MMAU-Pro)
                    audio_path_list = ann.get('audio_path_list', [])
                    if not audio_path_list:
                        continue
                    
                    paths = {
                        'audio_path_list': [self.process_path(os.path.join(self.data_prefix_dir, p)) for p in audio_path_list]
                    }
                    
                    # 检查所有音频文件是否存在
                    all_exist = True
                    for audio_path in paths['audio_path_list']:
                        if not os.path.exists(audio_path):
                            print(f"✗ 音频文件不存在: {audio_path}")
                            all_exist = False
                            break
                    if not all_exist:
                        continue
                    
                    # 使用第一个音频的时长进行检查
                    duration = ann.get('duration')
                    if duration is None:
                        duration = get_audio_duration(paths['audio_path_list'][0])
                else:
                    # 单音频格式 (其他数据集)
                    audio_path = ann.get('WavPath', '')
                    
                    paths = {
                        'audio_path': self.process_path(os.path.join(self.data_prefix_dir, audio_path))
                    }
                    if paths['audio_path'] and not os.path.exists(paths['audio_path']):
                        print(f"✗ 音频文件不存在: {paths['audio_path']}")
                        continue
                    
                    # 单音频时长检查
                    duration = ann.get('duration')
                    if duration is None and paths['audio_path'] and os.path.exists(paths['audio_path']):
                        duration = get_audio_duration(paths['audio_path'])
                
                # 记录音频时长统计
                self.duration_stats['total_count'] += 1
                if duration is not None:
                    self.duration_stats['durations'].append(duration)
                    
                    # 更新时长分布
                    if duration <= 10:
                        self.duration_stats['distribution']['0-10s'] += 1
                    elif duration <= 20:
                        self.duration_stats['distribution']['10-20s'] += 1
                    elif duration <= 30:
                        self.duration_stats['distribution']['20-30s'] += 1
                    elif duration <= 60:
                        self.duration_stats['distribution']['30-60s'] += 1
                    elif duration <= 120:
                        self.duration_stats['distribution']['60-120s'] += 1
                    else:
                        self.duration_stats['distribution']['>120s'] += 1
                    
                    # 可选：检查时长限制（如果设置了max_duration）
                    if self.max_duration is not None and duration > self.max_duration:
                        audio_display = paths.get('audio_path') or paths.get('audio_path_list', [''])[0]
                        print(f"✗ 音频文件超过{self.max_duration}秒 ({duration:.1f}s): {audio_display}")
                        continue
                # 如果无法读取时长（duration is None），仍然保留该样本
                # 支持更多文本字段名称
                text = (ann.get('text') or 
                       ann.get('sentence') or 
                       ann.get('transcription') or  # FLEURS
                       ann.get('label') or           # People's Speech
                       ann.get('normalized_text') or # VoxPopuli
                       ann.get('reference') or # VoiceBench bbh/sdqa/openbookqa/mmsu
                       ann.get('answer') or   # audio_trivia_qa, ClothoCaption
                       ann.get('Answer') or   # Voice CMMLU (大写A)
                       ann.get('answers') or  # audio_web_questions  
                       ann.get('caption') or  # AudioCaps
                       ann.get('gt') or       # MELD
                       '')
                
                # 处理prompt，支持question字段
                prompt = ann.get('prompt') or ann.get('question') or ''
                
                annotation = {
                    'prompt': prompt,
                    'gt_answer': text,
                    'dataset_name': self.dataset_name,
                    }
                
                # 只有 ifeval 数据集才添加 IFEvalMeta
                if 'ifeval' in self.dataset_name.lower():
                    annotation['IFEvalMeta'] = {
                        'instruction_id_list': ann.get('instruction_id_list', []),  # IFEval 指令类型列表
                        'kwargs': ann.get('kwargs', []),  # IFEval 详细参数
                        'key': ann.get('key', 0),  # IFEval 唯一标识
                    }
                
                # 如果存在 choices 字段，才添加
                # 注意：只有当 choices 列表长度 > 1 时才认为是有效的选择题
                if 'choices' in ann:
                    choices = ann['choices']
                    if isinstance(choices, list) and len(choices) > 1:
                        annotation['choices'] = choices
                    else:
                        # choices 列表长度 <= 1，不是有效的选择题，不添加 choices 字段，MMAU-Pro中存在这种数据
                        # 这样在推理时不会构造无效的 option prompts
                        pass
                
                # 如果存在 task 字段，才添加（如 MMAU 数据集）
                if 'task' in ann:
                    annotation['task'] = ann['task']
                
                # MMAU-Pro 特有字段
                if 'mmau_pro' in self.dataset_name.lower():
                    # 保存category字段用于区分评测类型
                    if 'category' in ann:
                        annotation['category'] = ann['category']
                    
                    # 指令遵循任务的特有字段
                    if ann.get('category') == 'instruction following':
                        annotation['MMAUProMeta'] = {
                            'task_identifier': ann.get('task_identifier', ''),
                            'kwargs': ann.get('kwargs', {}),
                            'transcription': ann.get('transcription', ''),
                        }
                
                data_list.append({
                    'paths': paths,
                    'annotation': annotation
                })
                self.duration_stats['valid_count'] += 1
        
        if unused_keys:
            print(f"  [Info] {self.dataset_name} 中未使用的字段: {sorted(unused_keys)}")
        
        # 打印音频时长分布统计
        self._print_duration_stats()
        
        return data_list
    
    def _print_duration_stats(self):
        """打印音频时长分布统计"""
        if self.duration_stats['total_count'] == 0:
            return
        
        print(f"\n{'='*60}")
        print(f"音频时长统计 - {self.dataset_name}")
        print(f"{'='*60}")
        print(f"总样本数: {self.duration_stats['total_count']}")
        print(f"有效样本数: {self.duration_stats['valid_count']}")
        
        if self.duration_stats['durations']:
            durations = self.duration_stats['durations']
            print(f"\n时长统计:")
            print(f"  平均时长: {np.mean(durations):.2f}s")
            print(f"  最短时长: {np.min(durations):.2f}s")
            print(f"  最长时长: {np.max(durations):.2f}s")
            print(f"  中位数: {np.median(durations):.2f}s")
            print(f"  标准差: {np.std(durations):.2f}s")
            
            print(f"\n时长分布:")
            total_with_duration = len(durations)
            for range_name, count in self.duration_stats['distribution'].items():
                percentage = (count / total_with_duration * 100) if total_with_duration > 0 else 0
                bar = '█' * int(percentage / 2)  # 每2%一个方块
                print(f"  {range_name:>10}: {count:>5} ({percentage:>5.1f}%) {bar}")
        else:
            print(f"\n⚠ 没有音频时长信息")
        
        print(f"{'='*60}\n")
    
    def process_path(self, path):
        """处理特定数据集的路径转换
        
        不同数据集可能需要不同的路径处理方式：
        - gigaspeech: 需要移除 '_metadata'
        - voicebench: WavPath中包含test/前缀，需要移除
        - meld: WavPath包含meld/前缀，需要移除
        - 其他数据集: 可以在这里添加特定的处理逻辑
        """
        # GigaSpeech 特殊处理
        if 'gigaspeech' in self.dataset_name.lower():
            path = path.replace('_metadata', '')
        
        # VoiceBench 特殊处理
        elif 'voicebench' in self.dataset_name.lower():
            # VoiceBench的WavPath包含'test/'前缀，但我们的软链接已经指向test目录
            # 所以需要移除这个前缀
            if 'test/' in path:
                path = path.replace('/test/', '/')
        
        # MELD 特殊处理
        elif 'meld' in self.dataset_name.lower():
            # MELD的路径中有 _wav 后缀，但实际目录没有这个后缀
            # output_repeated_splits_test_wav -> output_repeated_splits_test
            # dev_splits_complete_wav -> dev_splits_complete
            path = path.replace('_test_wav/', '_test/')
            path = path.replace('_complete_wav/', '_complete/')
        
        return path



def add_duration_to_jsonl(annotation_path, data_prefix_dir):
    """为JSONL文件添加duration字段
    
    Args:
        annotation_path: JSONL标注文件路径
        data_prefix_dir: 音频文件前缀目录
    
    Returns:
        bool: 是否更新了文件
    """
    import tempfile
    import shutil
    
    print(f"检查文件: {annotation_path}")
    
    # 先检查是否需要添加duration
    need_update = False
    with jsonlines.open(annotation_path, mode='r') as reader:
        for ann in reader:
            if 'duration' not in ann:
                need_update = True
                break
    
    if not need_update:
        print("  ✓ 已包含duration字段，无需更新")
        return False
    
    print("  ⚠ 缺少duration字段，开始计算音频时长...")
    
    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jsonl')
    os.close(temp_fd)
    
    updated_count = 0
    error_count = 0
    
    try:
        with jsonlines.open(annotation_path, mode='r') as reader:
            with jsonlines.open(temp_path, mode='w') as writer:
                for ann in reader:
                    # 如果已有duration，直接写入
                    if 'duration' in ann:
                        writer.write(ann)
                        continue
                    
                    # 计算音频时长
                    audio_path = ann.get('WavPath', '')
                    full_audio_path = os.path.join(data_prefix_dir, audio_path)
                    
                    if os.path.exists(full_audio_path):
                        duration = get_audio_duration(full_audio_path)
                        if duration is not None:
                            ann['duration'] = round(duration, 2)  # 保留两位小数
                            updated_count += 1
                        else:
                            error_count += 1
                    else:
                        error_count += 1
                    
                    writer.write(ann)
                    
                    # 每100个样本打印一次进度
                    if (updated_count + error_count) % 100 == 0:
                        print(f"    已处理 {updated_count + error_count} 个样本...")
        
        # 备份原文件并替换
        backup_path = annotation_path + '.backup'
        shutil.copy2(annotation_path, backup_path)
        shutil.move(temp_path, annotation_path)
        
        print(f"  ✓ 更新完成: {updated_count} 个音频添加了duration字段")
        if error_count > 0:
            print(f"  ⚠ {error_count} 个音频无法计算时长")
        print(f"  备份文件: {backup_path}")
        
        return True
        
    except Exception as e:
        # 出错时恢复原文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"  ✗ 更新失败: {e}")
        return False


if __name__ == '__main__':
    import sys
    
    # 显示帮助信息
    if '--help' in sys.argv or '-h' in sys.argv:
        print("音频数据集测试工具")
        print("\n用法:")
        print("  python -m o_e_Kit.datasets.audio_datasets [选项]")
        print("\n选项:")
        print("  --skip-duration    跳过自动添加duration字段的步骤")
        print("  --help, -h         显示此帮助信息")
        print("\n说明:")
        print("  默认情况下，程序会自动检查所有数据集，并为缺少duration字段的JSONL文件添加音频时长。")
        print("  这可以显著提高后续数据集加载的速度。")
        sys.exit(0)
    
    print("="*60)
    print("音频数据集加载测试")
    print("="*60)
    
    # 检查是否只想运行测试，不自动添加duration
    skip_duration = '--skip-duration' in sys.argv
    
    # 先测试已有的数据集
    test_configs = [
        # GigaSpeech (已有的)
        {
            'name': 'GigaSpeech Test',
            'annotation_path': './data/audio/asr/gigaspeech/test.jsonl',
            'data_prefix_dir': './data/audio/asr/gigaspeech/test_files/',
            'dataset_name': 'gigaspeech_test'
        },
        # WenetSpeech (已有的)
        {
            'name': 'WenetSpeech Test Net',
            'annotation_path': './data/audio/asr/wenetspeech/test_net.jsonl',
            'data_prefix_dir': './data/audio/asr/wenetspeech/test_net/',
            'dataset_name': 'wenetspeech_test_net'
        },
        {
            'name': 'WenetSpeech Test Meeting',
            'annotation_path': './data/audio/asr/wenetspeech/test_meeting.jsonl',
            'data_prefix_dir': './data/audio/asr/wenetspeech/test_meeting/',
            'dataset_name': 'wenetspeech_test_meeting'
        },
        # LibriSpeech test-clean
        {
            'name': 'LibriSpeech Test Clean',
            'annotation_path': './data/audio/asr/librispeech/test_clean.jsonl',
            'data_prefix_dir': './data/audio/asr/',
            'dataset_name': 'librispeech_test_clean'
        },
        # LibriSpeech test-other
        {
            'name': 'LibriSpeech Test Other',
            'annotation_path': './data/audio/asr/librispeech/test_other.jsonl',
            'data_prefix_dir': './data/audio/asr/',
            'dataset_name': 'librispeech_test_other'
        },
        # AISHELL-1 test
        {
            'name': 'AISHELL-1 Test',
            'annotation_path': './data/audio/asr/aishell1/test.jsonl',
            'data_prefix_dir': './data/audio/asr/aishell1/',
            'dataset_name': 'aishell1_test'
        },
        # AISHELL-2 test
        {
            'name': 'AISHELL-2 Test',
            'annotation_path': './data/audio/asr/aishell2/test.jsonl',
            'data_prefix_dir': './data/audio/asr/aishell2/',
            'dataset_name': 'aishell2_test'
        },
        # CommonVoice English
        {
            'name': 'CommonVoice English v15',
            'annotation_path': './data/audio/asr/commonvoice/en_v15_test.jsonl',
            'data_prefix_dir': './data/audio/asr/commonvoice/audios/',
            'dataset_name': 'commonvoice_en'
        },
        # CommonVoice Chinese
        {
            'name': 'CommonVoice Chinese v15',
            'annotation_path': './data/audio/asr/commonvoice/zh_v15_test.jsonl',
            'data_prefix_dir': './data/audio/asr/commonvoice/audios/',
            'dataset_name': 'commonvoice_zh'
        },
        # CommonVoice Cantonese
        {
            'name': 'CommonVoice Cantonese v15',
            'annotation_path': './data/audio/asr/commonvoice/yue_v15_test.jsonl',
            'data_prefix_dir': './data/audio/asr/commonvoice/audios/',
            'dataset_name': 'commonvoice_yue'
        },
        # CommonVoice French
        {
            'name': 'CommonVoice French v15',
            'annotation_path': './data/audio/asr/commonvoice/fr_v15_test.jsonl',
            'data_prefix_dir': './data/audio/asr/commonvoice/audios/',
            'dataset_name': 'commonvoice_fr'
        },
        # FLEURS 中文
        {
            'name': 'FLEURS 中文',
            'annotation_path': './data/audio/asr/fleurs/zh/test.jsonl',
            'data_prefix_dir': './data/audio/asr/fleurs/zh/',
            'dataset_name': 'fleurs_zh'
        },
        # FLEURS 英文
        {
            'name': 'FLEURS 英文',
            'annotation_path': './data/audio/asr/fleurs/en_us/test.jsonl',
            'data_prefix_dir': './data/audio/asr/fleurs/en_us/',
            'dataset_name': 'fleurs_en'
        },
        # People's Speech
        {
            'name': "People's Speech Test",
            'annotation_path': './data/audio/asr/peoples_speech/test.jsonl',
            'data_prefix_dir': './data/audio/asr/peoples_speech/test/test/',
            'dataset_name': 'peoples_speech_test'
        },
        # SPGISpeech
        {
            'name': 'SPGISpeech Test',
            'annotation_path': './data/audio/asr/spgispeech/test.jsonl',
            'data_prefix_dir': './data/audio/asr/spgispeech/',
            'dataset_name': 'spgispeech_test'
        },
        # TED-LIUM v3
        {
            'name': 'TED-LIUM v3 Test',
            'annotation_path': './data/audio/asr/tedlium/TEDLIUM_release3_test.jsonl',
            'data_prefix_dir': './data/audio/asr/tedlium/',
            'dataset_name': 'tedlium3_test'
        },
        # VoxPopuli 英文
        {
            'name': 'VoxPopuli English',
            'annotation_path': './data/audio/asr/voxpopuli/data/en/asr_test.jsonl',
            'data_prefix_dir': './data/audio/asr/voxpopuli/data/en/test/test_part_0/',
            'dataset_name': 'voxpopuli_en'
        },
        # VoiceBench AlpacaEval QA
        {
            'name': 'VoiceBench AlpacaEval',
            'annotation_path': './data/audio/qa/voicebench/alpacaeval_full/test.jsonl',
            'data_prefix_dir': './data/audio/qa/voicebench/',
            'dataset_name': 'voicebench_alpacaeval'
        },
        # Audio Web Questions
        {
            'name': 'Audio Web Questions',
            'annotation_path': './data/audio/qa/audio_web_questions/test_audio.jsonl',
            'data_prefix_dir': 'data/audio/qa',
            'dataset_name': 'audio_web_questions'
        },
        # Audio Trivia QA
        {
            'name': 'Audio Trivia QA',
            'annotation_path': './data/audio/qa/audio_trivia_qa/test-1024-clean.jsonl',
            'data_prefix_dir': 'data/audio/qa',
            'dataset_name': 'audio_trivia_qa'
        },
        # AudioCaps (Caption任务)
        {
            'name': 'AudioCaps',
            'annotation_path': './data/audio/caption/AudioCaps/test.jsonl',
            'data_prefix_dir': './data/audio/caption/AudioCaps/',
            'dataset_name': 'audiocaps_test'
        },
        # ClothoCaption (Caption任务)
        {
            'name': 'ClothoCaption',
            'annotation_path': './data/audio/caption/ClothoCaption/test.jsonl',
            'data_prefix_dir': './data/audio/caption/ClothoCaption/evaluation/',
            'dataset_name': 'clothocaption_test'
        },
        # WavCaps AudioSet_SL (Caption任务)
        {
            'name': 'WavCaps AudioSet_SL',
            'annotation_path': './data/audio/caption/WavCaps/AudioSet_SL_test.jsonl',
            'data_prefix_dir': './data/audio/caption/WavCaps/',
            'dataset_name': 'wavcaps_audioset_sl'
        },
        # WavCaps FreeSound (Caption任务)
        {
            'name': 'WavCaps FreeSound',
            'annotation_path': './data/audio/caption/WavCaps/FreeSound_test.jsonl',
            'data_prefix_dir': './data/audio/caption/WavCaps/',
            'dataset_name': 'wavcaps_freesound'
        },
        # WavCaps SoundBible (Caption任务)
        {
            'name': 'WavCaps SoundBible',
            'annotation_path': './data/audio/caption/WavCaps/SoundBible_test.jsonl',
            'data_prefix_dir': './data/audio/caption/WavCaps/',
            'dataset_name': 'wavcaps_soundbible'
        },
        # MMAU (多任务音频理解)
        {
            'name': 'MMAU Test Mini',
            'annotation_path': './data/audio/multitask/MMAU/test/test.jsonl',
            'data_prefix_dir': './data/audio/multitask/MMAU/',
            'dataset_name': 'mmau_test_mini'
        },
        # MMSU (多任务语音理解)
        {
            'name': 'MMSU Bench',
            'annotation_path': './data/audio/multitask/MMSU/test/test_lines_4996.jsonl',
            'data_prefix_dir': './data/audio/multitask/MMSU/',
            'dataset_name': 'mmsu_bench'
        },
        # MMAR (多模态音频推理)
        {
            'name': 'MMAR Bench',
            'annotation_path': './data/audio/multitask/MMAR/test/test.jsonl',
            'data_prefix_dir': './data/audio/multitask/MMAR/',
            'dataset_name': 'mmar_bench'
        },
        # VocalSound (音频分类任务)
        {
            'name': 'VocalSound',
            'annotation_path': './data/audio/cls/vocalsound/test.jsonl',
            'data_prefix_dir': './data/audio/cls/vocalsound/',
            'dataset_name': 'vocalsound'
        },
        # MELD (情感识别任务)
        {
            'name': 'MELD',
            'annotation_path': './data/audio/cls/MELD/meld_eval.jsonl',
            'data_prefix_dir': './data/audio/cls/MELD/',
            'dataset_name': 'meld'
        },
    ]
    
    # 首先检查并为所有缺少duration的数据集添加时长
    if not skip_duration:
        print("\n" + "="*60)
        print("检查并为缺少duration的数据集添加时长字段")
        print("="*60)
        
        datasets_need_update = []
        
        # 检查每个数据集是否需要添加duration
        for config in test_configs:
            if not os.path.exists(config['annotation_path']):
                continue
                
            # 检查是否需要添加duration
            need_update = False
            try:
                with jsonlines.open(config['annotation_path'], mode='r') as reader:
                    for ann in reader:
                        if 'duration' not in ann:
                            need_update = True
                            break
                
                if need_update:
                    datasets_need_update.append(config)
            except Exception as e:
                print(f"✗ 检查 {config['name']} 时出错: {e}")
        
        # 如果有数据集需要更新，询问是否处理
        if datasets_need_update:
            print(f"\n发现 {len(datasets_need_update)} 个数据集缺少duration字段：")
            for config in datasets_need_update:
                print(f"  - {config['name']}")
            
            print("\n正在自动添加duration字段...")
            
            for config in datasets_need_update:
                print(f"\n处理 {config['name']}...")
                add_duration_to_jsonl(config['annotation_path'], config['data_prefix_dir'])
            
            print("\n" + "="*60)
            print("Duration字段添加完成")
            print("="*60)
        else:
            print("\n✓ 所有数据集都已包含duration字段")
    else:
        print("\n跳过duration字段检查（使用了 --skip-duration 参数）")
    
    # 常规测试：测试每个数据集
    print("\n" + "="*60)
    print("开始测试数据集加载")
    print("="*60)
    
    for config in test_configs:
        print(f"\n测试 {config['name']}...")
        print("-"*40)
        
        # 检查标注文件是否存在
        if not os.path.exists(config['annotation_path']):
            print(f"✗ 标注文件不存在: {config['annotation_path']}")
            continue
        
        try:
            dataset = AudioEvalDataset(
                annotation_path=config['annotation_path'],
                data_prefix_dir=config['data_prefix_dir'],
                dataset_name=config['dataset_name']
            )
            
            print(f"✓ 成功加载，共 {len(dataset)} 个样本")
            
            # 显示前3个样本
            exist_count = 0
            for i in range(min(5, len(dataset))):
                idx, paths, annotation = dataset[i]
                print(f"\n  样本 {i+1}:")
                print(f"    音频路径: {paths['audio_path']}")
                # 处理 gt_answer 可能是列表的情况
                if isinstance(annotation['gt_answer'], list):
                    print(f"    答案列表: {annotation['gt_answer'][:3]}..." if len(annotation['gt_answer']) > 3 else f"    答案列表: {annotation['gt_answer']}")
                    if len(annotation['gt_answer']) > 1:
                        print(f"    (共{len(annotation['gt_answer'])}个答案)")
                else:
                    print(f"    文本内容: {annotation['gt_answer'][:60]}...")
                if annotation.get('prompt'):
                    print(f"    问题: {annotation['prompt'][:60]}...")
                if annotation.get('choices'):
                    print(f"    选项: {annotation['choices']}")
                
                # 检查音频文件是否存在
                if os.path.exists(paths['audio_path']):
                    print(f"    ✓ 音频文件存在")
                    exist_count += 1
                else:
                    print(f"    ✗ 音频文件不存在")
                    # 尝试找出问题
                    dir_path = os.path.dirname(paths['audio_path'])
                    file_name = os.path.basename(paths['audio_path'])
                    if os.path.exists(dir_path):
                        # 列出目录中类似的文件
                        import glob
                        similar_files = glob.glob(os.path.join(dir_path, file_name[:10] + "*"))[:3]
                        if similar_files:
                            print(f"    可能的文件: {[os.path.basename(f) for f in similar_files]}")
            
            print(f"\n  文件存在率: {exist_count}/{min(5, len(dataset))}")
                    
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)