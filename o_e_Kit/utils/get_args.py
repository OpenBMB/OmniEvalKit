"""
参数解析模块 - 整合所有参数配置
"""

import argparse
from o_e_Kit.utils.args import add_model_args, add_dataset_args, add_evaluation_flags, add_runtime_args
from o_e_Kit.utils.args.dataset_args import apply_evaluation_logic, get_dataset_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Duplex Model Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            参数说明：
            模型配置：包含模型路径、类型、生成参数等
            数据集配置：包含各种数据集的路径和评估标志
            运行时配置：包含系统设置、输出配置和评估控制参数

            使用示例：
            python eval_main.py --model_path /path/to/model --eval_gigaspeech_test --batchsize 4
            python eval_main.py --model_type minicpmo --eval_all_audio --max_sample_num 100
            python eval_main.py --eval_all --prefix experiment_1
        """
    )
    
    add_model_args(parser)
    add_dataset_args(parser)
    add_evaluation_flags(parser)
    add_runtime_args(parser)
    
    args = parser.parse_args()
    
    apply_evaluation_logic(args)
    
    args._parser = parser
    
    return args


def print_args(args):
    """格式化打印参数配置 - 使用argparse的分组信息"""
    print("🔧 参数配置")
    print("=" * 60)
    
    all_args = vars(args)

    parser = args._parser
    
    for group in parser._action_groups:
        group_params = {}
        for action in group._group_actions:
            if action.dest != 'help' and action.dest in all_args:
                value = all_args[action.dest]
                if value is not None:
                    group_params[action.dest] = value
        
        if group_params:
            title = group.title
            
            print(f"\n{title}:")
            for key, value in group_params.items():
                print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    
    print_args(parse_args())
    info = get_dataset_info()
    print(f"📊 注册的数据集 (总计: {info['total']} 个):")
    for category, configs in info['by_category'].items():
        print(f"  {category.upper()}: {[c.display_name for c in configs]}")
    print("\n" + "=" * 60)
    print("使用 --eval_all_audio 启用所有音频数据集")
    print("使用 --eval_all 启用所有数据集")