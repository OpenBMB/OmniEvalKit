"""运行时和系统配置参数"""

import argparse

def add_runtime_args(parser: argparse.ArgumentParser):
    """添加运行时和系统配置相关的参数"""
    
    # 系统执行设置
    system_group = parser.add_argument_group('系统配置', '硬件设备和执行环境相关参数')
    
    system_group.add_argument('--local-rank', type=int, default=0, 
                            help='分布式训练的本地rank')
    system_group.add_argument("--device", type=str, default="cuda:0",
                            help="计算设备")
    system_group.add_argument('--batchsize', type=int, default=1, 
                            help='处理的批次大小')
    
    # 输出和保存设置
    output_group = parser.add_argument_group('输出配置', '结果保存和输出相关参数')
    
    output_group.add_argument("--answer_path", type=str, default="./answers",
                            help="答案保存路径")
    output_group.add_argument("--prefix", type=str, default="",
                            help="输出文件前缀")
    
    # 评估控制参数
    eval_control_group = parser.add_argument_group('评估设置', '评估过程控制参数')
    
    eval_control_group.add_argument("--max_sample_num", type=int, default=None, 
                                  help="最大评估样本数量")
    eval_control_group.add_argument("--shuffle", action="store_true",
                                  help="从全量数据中随机采样 max_sample_num 条")
    eval_control_group.add_argument("--shuffle_after_limit", action="store_true",
                                  help="先取前 max_sample_num 条，再随机打乱顺序")
    eval_control_group.add_argument("--shuffle_seed", type=int, default=42,
                                  help="随机采样种子，用于可复现")
    eval_control_group.add_argument("--route_idx", type=int, default=0,
                                  help="多路并行评测时的当前路编号（从0开始）")
    eval_control_group.add_argument("--route_num", type=int, default=1,
                                  help="多路并行评测的总路数，用于在job内部手动划分数据子集")

def get_runtime_args():
    """获取仅包含运行时参数的解析器（用于测试）"""
    parser = argparse.ArgumentParser(description="运行时参数配置")
    add_runtime_args(parser)
    return parser.parse_args()

if __name__ == "__main__":
    # 测试运行时参数
    args = get_runtime_args()
    print("运行时参数配置:")
    
    # 按功能分组打印参数
    all_args = vars(args)
    
    system_args = {k: v for k, v in all_args.items() 
                   if k in ['local_rank', 'device', 'batchsize']}
    output_args = {k: v for k, v in all_args.items() 
                   if k in ['answer_path', 'prefix']}
    eval_args = {k: v for k, v in all_args.items() 
                 if k in ['max_sample_num', 'shuffle', 'shuffle_seed', 'route_idx', 'route_num']}
    
    print("\n🖥️  系统配置:")
    for key, value in system_args.items():
        print(f"  {key}: {value}")
    
    print("\n💾 输出配置:")
    for key, value in output_args.items():
        print(f"  {key}: {value}")
    
    print("\n⚙️  评估控制:")
    for key, value in eval_args.items():
        print(f"  {key}: {value}")