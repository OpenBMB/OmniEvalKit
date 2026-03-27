"""
omnievalkit评估主程序

支持多种数据集的评估，包括：
- 音频数据集：GigaSpeech、WeNetSpeech、AudioQA1M等
- 视频数据集：OVOBench、StreamingBench
- 多模态数据集：VisionCap、OmniCap、LiveCC、AVEvent等

StreamingBench支持四种任务类型：
1. real: 实时视觉理解 - 基于当前时刻的视觉问答
2. omni: 全模态理解 - 结合视频和音频的多模态问答  
3. sqa: 序列问答 - 基于视频历史的连续对话
4. proactive: 主动输出 - 特殊的循环推理逻辑，模型主动判断输出时机

proactive任务的完整适配：
- 推理阶段：使用generate_proactive方法，实现循环时间判断 + 两阶段问答
- 评估阶段：使用StreamingProactiveEval评估器，考虑时间和内容双重准确性

使用示例：
python eval_main.py --model_type minicpmo_chat --eval_streamingbench --streamingbench_tasks proactive
python eval_main.py --eval_streamingbench --streamingbench_tasks real omni sqa proactive
"""

import torch
import numpy as np
import os
import datetime

from o_e_Kit.utils.get_args import parse_args
from o_e_Kit.utils.model_loader import load_model
from o_e_Kit.utils.evaluation_runner import run_all_evaluations, save_evaluation_results


def main(args):
    """主函数：初始化环境、加载模型、运行评估"""
    
    # 设置时间戳
    if args.prefix == "":
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        time = args.prefix
    
    # 初始化分布式训练
    # 增加 NCCL 超时时间，避免因推理速度不均匀导致的超时
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(hours=2),  # 设置 2 小时超时
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    print(f'Init Rank-{torch.distributed.get_rank()}')
    
    if torch.distributed.is_initialized():
        args.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # 加载模型
    model = load_model(args, args.device)
    
    # 运行所有评估
    result = run_all_evaluations(args, model, args.device, time)
    
    # 同步所有进程
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # 只在主进程保存结果
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    # 保存评估结果
    save_evaluation_results(result, args, time)
    
    # 清理分布式训练
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    main(args)