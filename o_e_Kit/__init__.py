"""
o_e_Kit 包 - 多模态评估工具包

主要模块:
- datasets: 数据集类 (AudioEvalDataset, OmniEvalDataset, BaseDataset)
- models: 模型类
- utils: 工具函数 (参数解析, 数据加载, 推理等)
"""

__version__ = "1.0.0"

# 导入主要的数据集类
from .datasets import AudioEvalDataset, OmniEvalDataset, BaseDataset

__all__ = [
    "AudioEvalDataset",
    "OmniEvalDataset", 
    "BaseDataset",
]