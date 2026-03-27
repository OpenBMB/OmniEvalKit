"""配置文件加载工具"""
import json
import os
from typing import Optional


def flatten_config(config: dict) -> dict:
    """
    递归展平嵌套配置，提取所有叶子节点（包含具体配置的字典）
    
    叶子节点判断：包含 'user_prompt' 或 'max_tokens' 等配置字段
    
    Args:
        config: 嵌套的配置字典
        
    Returns:
        展平后的字典 {dataset_name: config}
    """
    merged = {}
    
    def is_leaf_config(d: dict) -> bool:
        """判断是否是叶子配置节点"""
        config_keys = {'user_prompt', 'system_prompt', 'max_tokens', 'max_frames'}
        return bool(config_keys & set(d.keys()))
    
    def recurse(d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                if is_leaf_config(value):
                    # 叶子节点，key 是数据集名称
                    merged[key] = value
                else:
                    # 继续递归
                    recurse(value)
    
    recurse(config)
    return merged


def load_config(config_path: str, flatten: bool = True) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        flatten: 是否展平配置，默认 True
        
    Returns:
        如果 flatten=True，返回展平后的配置字典 {dataset_name: config}
        如果 flatten=False，返回原始配置字典
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Config file {config_path} not found")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
    
    if flatten:
        return flatten_config(full_config)
    return full_config

