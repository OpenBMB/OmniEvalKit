"""
数据集基类模块，支持多模态数据处理。

设计理念:
- 统一的数据访问接口: 所有数据集都返回 (idx, paths, annotation) 三元组
- 模态化设计: 支持 audio/video/omni 三种模态，每种模态有不同的路径要求
- 灵活的标注结构: annotation 字典可以包含任意任务相关的元数据
"""

import os
import abc
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

class BaseDataset(Dataset, abc.ABC):
    """
    数据集抽象基类，支持多模态数据处理。
    
    模态定义:
    - "audio": 纯音频数据，paths 必须包含 "audio_path" (单音频) 或 "audio_path_list" (多音频)
    - "video": 纯视频数据，paths 必须包含 "video_path" 
    - "omni": 多模态数据，paths 可以包含 "audio_path"/"audio_path_list" 和/或 "video_path"
    
    Args:
        annotation_path: 标注文件路径
        data_prefix_dir: 数据文件根目录
        modal: 数据模态，可选 ["audio", "video", "omni"]
    
    Returns:
        三元组 (idx, paths, annotation)，其中 paths 根据模态包含不同的路径字段
    """
    
    SUPPORTED_MODALS = ["audio", "video", "omni"]
    
    def __init__(self, annotation_path: str, data_prefix_dir: str = "", modal: str = "omni"):
        """
        初始化数据集
        
        Args:
            annotation_path: 标注文件路径
            data_prefix_dir: 数据文件根目录
            modal: 数据模态 ["audio", "video", "omni"]
        """
        if modal not in self.SUPPORTED_MODALS:
            raise ValueError(f"不支持的模态: {modal}，支持的模态: {self.SUPPORTED_MODALS}")
            
        self.annotation_path = annotation_path
        self.data_prefix_dir = data_prefix_dir
        self.modal = modal
        
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
        
        self.data = self._load_annotations()
        
        self._validate_paths()
        
        print(f"数据集初始化完成: {len(self.data)} 个样本, 模态: {modal}")

    def _validate_paths(self):
        for i, item in enumerate(self.data):
            paths = item.get("paths", {})
            
            if self.modal == "audio":
                # 支持单音频（audio_path）和多音频（audio_path_list）两种格式
                if "audio_path" not in paths and "audio_path_list" not in paths:
                    raise ValueError(f"音频模态数据集样本 {i} 缺少 audio_path 或 audio_path_list")
            elif self.modal == "video":
                if "video_path" not in paths:
                    raise ValueError(f"视频模态数据集样本 {i} 缺少 video_path")
            elif self.modal == "omni":
                # omni 模态支持两种路径格式：
                # 格式1: audio_path, video_path, image_path
                # 格式2: audio_paths_dict, image_paths_dict, video_paths_dict (字典)
                has_format1 = "audio_path" in paths or "audio_path_list" in paths or "video_path" in paths or "image_path" in paths
                has_format2 = "audio_paths_dict" in paths or "image_paths_dict" in paths or "video_paths_dict" in paths
                if not has_format1 and not has_format2:
                    raise ValueError(f"多模态数据集样本 {i} 缺少路径信息")

    @abc.abstractmethod
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        抽象方法：加载标注数据
        
        子类必须实现此方法，返回格式为:
        [
            {
                "paths": {
                    "audio_path": "...",          # 单音频路径
                    # 或
                    "audio_path_list": [...],     # 多音频路径列表（用于MMAU-Pro等数据集）
                    # 和/或
                    "video_path": "..."           # 视频路径
                },
                "annotation": {"gt_answer": "...", "task_type": "...", ...}
            },
            ...
        ]
        """
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
        if idx >= len(self.data):
            raise IndexError(f"索引 {idx} 超出数据集大小 {len(self.data)}")
        
        sample_info = self.data[idx]
        paths = sample_info.get("paths", {})
        annotation = sample_info.get("annotation", {})
        
        return idx, paths, annotation