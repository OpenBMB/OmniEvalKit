"""参数配置模块包

该包将参数配置分为三个模块：
- model_args: 模型相关参数（路径、类型、生成配置等）
- dataset_args: 数据集相关参数（路径、评估标志等）
- runtime_args: 运行时配置参数（系统设置、输出配置等）
"""

from .model_args import add_model_args
from .dataset_args import add_dataset_args, add_evaluation_flags  
from .runtime_args import add_runtime_args

__all__ = [
    'add_model_args',
    'add_dataset_args', 
    'add_evaluation_flags',
    'add_runtime_args'
]