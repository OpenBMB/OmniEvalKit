"""多模态数据集模块，处理音频+视频的多模态数据"""

import os
import re
import subprocess
from typing import List, Dict, Any
import jsonlines
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("Warning: cv2 is not installed. Frame extraction will not be available.")
    cv2 = None


from torch.utils.data import Dataset
from o_e_Kit.datasets.base_dataset import BaseDataset


class OmniEvalDataset(BaseDataset):
    """
    统一多模态评估数据集
    
    统一标注格式:
    - dataset_type: 任务类型 (mcq, open_qa, caption)
    - dataset_name: 数据来源 (daily_omni, omnibench, unobench, worldsense)
    - choices: MCQ选项 ["A. 内容", "B. 内容", ...]
    - gt_answer: MCQ为字母 "A", 开放问答为完整内容
    """
    
    def __init__(self, annotation_path: str, data_prefix_dir: str = "", 
                 dataset_name: str = '', auto_extract_audio: bool = False):
        """
        Args:
            annotation_path: JSONL 标注文件路径
            data_prefix_dir: 数据根目录
            dataset_name: 数据集名称（覆盖 jsonl 中的 dataset_name）
            auto_extract_audio: 是否自动从视频提取音频（如果缺失）
        """
        self.dataset_name = dataset_name
        self.auto_extract_audio = auto_extract_audio
        super().__init__(annotation_path, data_prefix_dir, modal="omni")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        data_list = []
        
        with jsonlines.open(self.annotation_path, mode='r') as reader:
            for ann in reader:
                paths = self._extract_paths(ann)
                
                # 自动提取音频
                if self.auto_extract_audio and 'video_path' in paths and 'audio_path' not in paths:
                    audio_path = self._extract_audio_if_needed(paths['video_path'])
                    if audio_path:
                        paths['audio_path'] = audio_path
                
                # 跳过没有任何媒体路径的样本
                if not paths:
                    continue
                
                annotation = self._build_annotation(ann)
                
                data_list.append({
                    'paths': paths,
                    'annotation': annotation
                })
        
        return data_list
    
    def _extract_paths(self, ann: Dict) -> Dict[str, Any]:
        """
        提取路径信息，支持两种格式：
        
        格式1 (Daily-Omni, OmniBench, WorldSense):
            {'audio_path': '...', 'video_path': '...', 'image_path': '...'}
        
        格式2 (UNO-Bench):
            {'audio_paths': {...}, 'image_paths': {...}, 'video_paths': {...}}
        """
        paths = {}
        
        # 格式1: 直接路径字段 (WavPath, VideoPath, ImagePath)
        wav_path = ann.get('WavPath', '')
        video_path = ann.get('VideoPath', '')
        image_path = ann.get('ImagePath', '')
        
        if wav_path or video_path or image_path:
            if wav_path:
                paths['audio_path'] = self.process_path(os.path.join(self.data_prefix_dir, wav_path))
            if video_path:
                paths['video_path'] = os.path.join(self.data_prefix_dir, video_path)
            if image_path:
                paths['image_path'] = os.path.join(self.data_prefix_dir, image_path)
            return paths
        
        # 格式2: 路径字典 (audio_paths_dict, image_paths_dict, video_paths_dict)
        audio_paths = ann.get('audio_paths_dict', {})
        image_paths = ann.get('image_paths_dict', {})
        video_paths = ann.get('video_paths_dict', {})
        
        if audio_paths:
            # 添加 data_prefix_dir 前缀
            paths['audio_paths_dict'] = {k: os.path.join(self.data_prefix_dir, v) for k, v in audio_paths.items() if v}
        if image_paths:
            paths['image_paths_dict'] = {k: os.path.join(self.data_prefix_dir, v) for k, v in image_paths.items() if v}
        if video_paths:
            paths['video_paths_dict'] = {k: os.path.join(self.data_prefix_dir, v) for k, v in video_paths.items() if v}
        
        return paths
    
    def _build_annotation(self, ann: Dict) -> Dict[str, Any]:
        """构建标注信息 - 统一格式"""
        dataset_type = ann.get('dataset_type', 'mcq')
        dataset_name = self.dataset_name or ann.get('dataset_name', '')
        
        # 基础字段
        annotation = {
            'dataset_type': dataset_type,
            'dataset_name': dataset_name,
            'question': ann.get('question', ''),
            'gt_answer': ann.get('gt_answer', ''),
        }
        
        # MCQ 类型: 添加 choices
        if dataset_type == 'mcq':
            annotation['choices'] = ann.get('choices', [])
        
        # 开放问答: 可能有评分相关字段
        elif dataset_type == 'open_qa':
            annotation['gt_answer_content'] = ann.get('gt_answer_content', '')
            annotation['score_type'] = ann.get('score_type', 0)
        
        # Caption 类型
        elif dataset_type == 'caption':
            annotation['event_title'] = ann.get('event_title', '')
            annotation['preasr_text'] = ann.get('preasr_text', '')
        
        # 添加所有元数据字段（排除已处理的基础字段）
        base_fields = {'dataset_type', 'dataset_name', 'question', 'gt_answer', 
                       'choices', 'WavPath', 'VideoPath', 'ImagePath'}
        for key, value in ann.items():
            if key not in base_fields and value is not None:
                annotation[key] = value
        
        return annotation
    
    def _extract_audio_if_needed(self, video_path: str) -> str:
        """从视频提取音频（如果需要）"""
        if not os.path.exists(video_path):
            return ''
        
        # 生成音频路径
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(video_dir, f"{video_name}.wav")
        
        if os.path.exists(audio_path):
            return audio_path
        
        # 使用 ffmpeg 提取音频
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            if result.returncode == 0:
                return audio_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return ''
    
    def process_path(self, path):
        return path


class omni_datasetWoann(Dataset):
    """
    原有的omni数据集类，用于视频处理和帧提取。
    保持向后兼容性。
    """
    
    def __init__(self, data_prefix, video_type_list=None):
        self.data_prefix = data_prefix
        if video_type_list is None:
            self.video_type_list = [
                '幻觉类-tianyu', '推理类-tianchi', '详细描述类-tianchi',
                '推理类-tianyu', '详细描述类-tianyu'
            ]
        else:
            self.video_type_list = video_type_list
        
        self.video_path_list = []
        
        for video_type in self.video_type_list:
            video_type_dir = os.path.join(self.data_prefix, video_type)
            if os.path.exists(video_type_dir):
                for video_file in os.listdir(video_type_dir):
                    if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_path = os.path.join(video_type_dir, video_file)
                        self.video_path_list.append({
                            'video_path': video_path,
                            'video_type': video_type
                        })
    
    def __len__(self):
        return len(self.video_path_list)
    
    def __getitem__(self, idx):
        return {
            'idx': idx,
            'video_path': self.video_path_list[idx]['video_path'],
            'video_type': self.video_path_list[idx]['video_type']
        }
    
    def to_frame_and_mp3(self, save_dir, frame_rate=1):
        """将视频转换为帧和音频文件"""
        if cv2 is None:
            raise ImportError("OpenCV is required for video processing. Please install it with 'pip install opencv-python'.")
        
        for idx, item in enumerate(tqdm(self.video_path_list, desc="Processing videos")):
            video_path = item['video_path']
            output_dir = os.path.join(save_dir, str(idx))
            frame_dir = os.path.join(output_dir, 'input_frame_dir')
            os.makedirs(frame_dir, exist_ok=True)
            
            # 提取帧
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / frame_rate))
            
            frame_count = 0
            saved_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(frame_dir, f'frame_{saved_frame_count:06d}.jpg')
                    cv2.imwrite(frame_filename, frame)
                    saved_frame_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # 提取音频
            audio_path = os.path.join(output_dir, 'input_audio.mp3')
            command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '0', '-y', audio_path]
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"Error extracting audio from {video_path}:\n{e.stderr.decode()}")
            except FileNotFoundError:
                tqdm.write("Error: ffmpeg command not found. Please make sure ffmpeg is installed and in your PATH.")
                return
        
        print(f"Processing complete. Data saved in {save_dir}")


if __name__ == '__main__':
    import sys
    
    # 显示帮助信息
    if '--help' in sys.argv or '-h' in sys.argv:
        print("多模态数据集测试工具")
        print("\n用法:")
        print("  python -m o_e_Kit.datasets.omni_datasets [选项]")
        print("\n选项:")
        print("  --help, -h         显示此帮助信息")
        print("\n说明:")
        print("  测试所有已注册的多模态数据集，检查数据完整性。")
        sys.exit(0)
    
    print("="*60)
    print("多模态数据集加载测试 (统一格式)")
    print("="*60)
    
    # 测试配置列表
    test_configs = [
        # {
        #     'name': 'Daily-Omni',
        #     'annotation_path': './data/omni/raw_hf/daily-omni/daily_omni.jsonl',
        #     'data_prefix_dir': './data/omni/raw_hf/daily-omni/',
        #     'dataset_name': 'daily_omni'
        # },
        # {
        #     'name': 'OmniBench',
        #     'annotation_path': './data/omni/raw_hf/omnibench/omnibench.jsonl',
        #     'data_prefix_dir': './data/omni/raw_hf/omnibench/',
        #     'dataset_name': 'omnibench'
        # },
        # {
        #     'name': 'UNO-Bench',
        #     'annotation_path': './data/omni/raw_hf/uno-bench/unobench.jsonl',
        #     'data_prefix_dir': './data/omni/raw_hf/uno-bench/',
        #     'dataset_name': 'unobench'
        # },
        {
            'name': 'UNO-Bench MCQ',
            'annotation_path': './data/omni/raw_hf/uno-bench/unobench_mc.jsonl',
            'data_prefix_dir': './data/omni/raw_hf/uno-bench/',
            'dataset_name': 'unobench_mc'
        },
        # {
        #     'name': 'WorldSense',
        #     'annotation_path': './data/omni/raw_hf/worldsense/worldsense.jsonl',
        #     'data_prefix_dir': './data/omni/raw_hf/worldsense/',
        #     'dataset_name': 'worldsense'
        # },
        # {
        #     'name': 'AV-Odyssey',
        #     'annotation_path': './data/omni/raw_hf/av-odyssey/av_odyssey.jsonl',
        #     'data_prefix_dir': './data/omni/raw_hf/av-odyssey/',
        #     'dataset_name': 'av_odyssey'
        # },
        # {
        #     'name': 'LiveCC-Bench QA',
        #     'annotation_path': './data/omni/livecc-bench/livecc_bench_qa.jsonl',
        #     'data_prefix_dir': './data/omni/livecc-bench/',
        #     'dataset_name': 'livecc_bench_qa'
        # },
        # {
        #     'name': 'LiveSports-3K CC',
        #     'annotation_path': './data/omni/livecc-bench/livecc_bench_cc.jsonl',
        #     'data_prefix_dir': './data/omni/livecc-bench/',
        #     'dataset_name': 'livesports3k_cc'
        # },
        # {
        #     'name': 'Video-Holmes',
        #     'annotation_path': './data/omni/video-holmes/video_holmes.jsonl',
        #     'data_prefix_dir': './data/omni/video-holmes/',
        #     'dataset_name': 'video_holmes'
        # },
        {
            'name': 'Video-MME',
            'annotation_path': './data/omni/raw_hf/videomme/videomme.jsonl',
            'data_prefix_dir': './data/omni/raw_hf/videomme/',
            'dataset_name': 'videomme'
        },
    ]
    
    # 测试每个数据集
    for config in test_configs:
        print(f"\n测试 {config['name']}...")
        print("-"*40)
        
        # 检查标注文件是否存在
        if not os.path.exists(config['annotation_path']):
            print(f"✗ 标注文件不存在: {config['annotation_path']}")
            continue
        
        try:
            dataset = OmniEvalDataset(
                annotation_path=config['annotation_path'],
                data_prefix_dir=config['data_prefix_dir'],
                dataset_name=config['dataset_name']
            )
            
            print(f"✓ 成功加载，共 {len(dataset)} 个样本")
            
            # 统计 dataset_type 分布
            type_counts = {}
            for i in range(len(dataset)):
                _, _, ann = dataset[i]
                t = ann.get('dataset_type', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
            
            print(f"\n  dataset_type 分布:")
            for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"    {t}: {count}")
            
            # 显示前2个样本
            for i in range(min(2, len(dataset))):
                idx, paths, annotation = dataset[i]
                print(f"\n  样本 {i+1}:")
                
                # 显示路径
                for path_type, path_value in paths.items():
                    if isinstance(path_value, dict):
                        # dict 格式路径 (UNO-Bench, AV-Odyssey)
                        print(f"    {path_type}: [{len(path_value)} 个]")
                        for k, v in list(path_value.items())[:2]:
                            exists = "✓" if os.path.exists(v) else "✗"
                            print(f"      {k}: .../{os.path.basename(v)} [{exists}]")
                    else:
                        # 单一路径
                        exists = "✓" if os.path.exists(path_value) else "✗"
                        print(f"    {path_type}: .../{os.path.basename(path_value)} [{exists}]")
                
                # 显示关键字段
                print(f"    dataset_type: {annotation.get('dataset_type')}")
                print(f"    question: {annotation.get('question', '')[:40]}...")
                
                if annotation.get('dataset_type') == 'mcq':
                    choices = annotation.get('choices', [])
                    print(f"    choices: [{len(choices)} 个选项]")
                    print(f"    gt_answer: {annotation.get('gt_answer')}")
                else:
                    print(f"    gt_answer: {str(annotation.get('gt_answer', ''))[:40]}...")
                    
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
