#!/usr/bin/env python3
"""
简单统一的进度显示工具
自动处理 tqdm 兼容性问题，提供一致的接口
"""

import time
import logging
import sys
import os
from typing import Iterator, Any, Optional
import torch

# 环境配置
PROGRESS_CONFIG = {
    "log_interval": int(os.getenv("PROGRESS_LOG_INTERVAL", "10")),
    "force_cloud": os.getenv("PROGRESS_FORCE_CLOUD", "false").lower() == "true",
    "enable_emojis": os.getenv("PROGRESS_ENABLE_EMOJIS", "true").lower() == "true",
    "log_format": os.getenv("PROGRESS_LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
}

def smart_progress(iterable: Iterator[Any], 
                  desc: str = "Processing",
                  log_interval: Optional[int] = None,
                  force_cloud: Optional[bool] = None) -> Iterator[Any]:
    """
    智能进度显示，自动选择最佳方式
    
    Args:
        iterable: 可迭代对象
        desc: 描述信息
        log_interval: 云端模式下的日志间隔（None时使用环境配置）
        force_cloud: 强制使用云端友好模式（None时使用环境配置）
    
    Returns:
        迭代器
    
    Environment Variables:
        PROGRESS_LOG_INTERVAL: 默认日志间隔 (默认: 10)
        PROGRESS_FORCE_CLOUD: 强制云端模式 (默认: false)
        PROGRESS_ENABLE_EMOJIS: 启用emoji (默认: true)
        PROGRESS_LOG_FORMAT: 日志格式 (默认: %(asctime)s - %(levelname)s - %(message)s)
    """
    # 使用环境配置作为默认值
    log_interval = log_interval if log_interval is not None else PROGRESS_CONFIG["log_interval"]
    force_cloud = force_cloud if force_cloud is not None else PROGRESS_CONFIG["force_cloud"]
    
    # 检测是否强制使用云端模式
    if force_cloud:
        return _cloud_progress(iterable, desc, log_interval)
    
    # 检测运行环境
    if _is_cloud_environment():
        return _cloud_progress(iterable, desc, log_interval)
    
    # 尝试使用tqdm
    try:
        from tqdm import tqdm
        # 检查是否在交互式终端
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return tqdm(iterable, desc=desc)
        else:
            # 非交互式环境，使用云端模式
            return _cloud_progress(iterable, desc, log_interval)
    except ImportError:
        # tqdm不可用，使用云端模式
        return _cloud_progress(iterable, desc, log_interval)

def _is_cloud_environment() -> bool:
    """检测是否在云端环境"""
    cloud_indicators = [
        'SLURM_JOB_ID',      # SLURM集群
        'PBS_JOBID',         # PBS集群  
        'LSB_JOBID',         # LSF集群
        'KUBERNETES_SERVICE_HOST',  # Kubernetes
        'DOCKER_CONTAINER',  # Docker容器
    ]
    return any(os.getenv(var) for var in cloud_indicators)

def _cloud_progress(iterable: Iterator[Any], 
                   desc: str = "Processing",
                   log_interval: int = 10) -> Iterator[Any]:
    """云端友好的进度显示"""
    logger = logging.getLogger(__name__)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # 如果logger没有handler，添加默认的
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(PROGRESS_CONFIG["log_format"])
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # emoji配置
    use_emojis = PROGRESS_CONFIG["enable_emojis"]
    start_emoji = "🚀 " if use_emojis else ""
    progress_emoji = "📊 " if use_emojis else ""
    complete_emoji = "✅ " if use_emojis else ""
    
    # 尝试获取总长度
    total = len(iterable) if hasattr(iterable, '__len__') else None
    
    if total is not None:
        # 有总长度的情况
        logger.info(f"rank {rank} {start_emoji}开始 {desc}: 总共 {total} 个项目")
        
        start_time = time.time()
        for i, item in enumerate(iterable, 1):
            if i % log_interval == 0 or i == total or i == 1:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                progress_pct = (i / total) * 100
                
                if i == total:
                    logger.info(f"rank {rank} {complete_emoji}{desc} 完成: {i}/{total} (100.0%) 耗时: {elapsed:.1f}s, 平均速度: {rate:.2f}it/s")
                else:
                    eta = (total - i) / rate if rate > 0 else 0
                    logger.info(f"rank {rank} {progress_emoji}{desc} 进度: {i}/{total} ({progress_pct:.1f}%) 速度: {rate:.2f}it/s, 预计剩余: {eta:.0f}s")
            
            yield item
    else:
        # 未知长度的情况
        logger.info(f"rank {rank} {start_emoji}开始 {desc}")
        count = 0
        start_time = time.time()
        
        for item in iterable:
            count += 1
            if count % log_interval == 0 or count == 1:
                elapsed = time.time() - start_time
                rate = count / elapsed if elapsed > 0 else 0
                logger.info(f"rank {rank} {progress_emoji}{desc} 已处理: {count} 项, 速度: {rate:.2f}it/s")
            yield item
        
        elapsed = time.time() - start_time
        rate = count / elapsed if elapsed > 0 else 0
        logger.info(f"rank {rank} {complete_emoji}{desc} 完成: 总共处理 {count} 项, 耗时: {elapsed:.1f}s, 平均速度: {rate:.2f}it/s")

# 提供简单的别名，便于替换
def progress_bar(iterable, desc="Processing", **kwargs):
    """简单的进度条接口，自动适配环境"""
    return smart_progress(iterable, desc=desc, **kwargs)

# 兼容tqdm的接口
def tqdm_compatible(iterable, desc="Processing", **kwargs):
    """
    兼容tqdm的接口
    自动过滤不支持的参数，提供统一体验
    """
    # 只保留我们支持的参数
    supported_kwargs = {}
    if 'log_interval' in kwargs:
        supported_kwargs['log_interval'] = kwargs['log_interval']
    if 'force_cloud' in kwargs:
        supported_kwargs['force_cloud'] = kwargs['force_cloud']
    
    return smart_progress(iterable, desc=desc, **supported_kwargs)

# 配置函数
def set_progress_config(**kwargs):
    """
    设置进度显示配置
    
    Args:
        log_interval: 日志间隔
        force_cloud: 强制云端模式
        enable_emojis: 启用emoji
        log_format: 日志格式
    """
    for key, value in kwargs.items():
        if key in PROGRESS_CONFIG:
            PROGRESS_CONFIG[key] = value
        else:
            print(f"Warning: Unknown config key '{key}'. Available keys: {list(PROGRESS_CONFIG.keys())}")

def get_progress_config():
    """获取当前进度显示配置"""
    return PROGRESS_CONFIG.copy()

if __name__ == "__main__":
    print("🧪 测试智能进度显示")
    print("=" * 40)
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 显示当前配置
    print("当前环境配置:")
    config = get_progress_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 测试1: 默认配置
    print("1. 默认配置测试:")
    for item in smart_progress(range(20), desc="默认配置"):
        time.sleep(0.05)
    
    # 测试2: 修改配置
    print("\n2. 修改配置测试:")
    set_progress_config(log_interval=3, enable_emojis=False)
    for item in smart_progress(range(15), desc="修改后配置"):
        time.sleep(0.05)
    
    # 测试3: 临时覆盖
    print("\n3. 临时覆盖测试:")
    for item in smart_progress(range(10), desc="临时覆盖", log_interval=2):
        time.sleep(0.05)
    
    print("\n✅ 测试完成！")
    print("\n💡 环境变量设置示例:")
    print("export PROGRESS_LOG_INTERVAL=5")
    print("export PROGRESS_FORCE_CLOUD=true") 
    print("export PROGRESS_ENABLE_EMOJIS=false") 