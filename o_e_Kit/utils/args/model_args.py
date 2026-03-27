"""模型相关参数配置"""

import argparse

def add_model_args(parser: argparse.ArgumentParser):
    """添加模型相关的参数"""
    
    model_group = parser.add_argument_group('模型配置', '模型路径、类型和生成相关参数')
    
    model_group.add_argument("--model_path", type=str, default="path/to/your/modeling", 
                           help="模型检查点目录路径")
    model_group.add_argument("--config_path", type=str, default=None, 
                           help="模型配置文件路径")
    model_group.add_argument("--tokenizer_path", type=str, default='path/to/your/tokenizer',
                           help="分词器路径")
    model_group.add_argument("--pt_path", type=str, default=None, 
                           help="具体的.pt检查点文件路径")

    model_group.add_argument("--ls_mode", type=str, default='explicit', 
                           choices=['explicit', 'implicit'],
                           help="LS流式生成模式")
    model_group.add_argument("--model_name", type=str, default="minicpm26o",
                           help="模型名称")
    model_group.add_argument("--model_type", type=str, default='minicpmo', 
                           choices=[
                               'minicpmo',              # MiniCPM-O 统一模型
                               'minicpmo_duplex_demo',  # MiniCPM-O Duplex Demo
                               'whisper',               # Whisper ASR
                               'qwen3_omni',            # Qwen3-Omni 多模态理解模型
                               'gemini_omni',           # Gemini 多模态 API 评测模型
                           ],
                           help="使用的模型逻辑类型")
    
    generation_group = parser.add_argument_group('生成配置', '模型生成和解码相关参数')
    
    generation_group.add_argument("--generate_method", type=str, default=None,
                                choices=["batch", "chat", "generate"],
                                help=("推理方法："
                                      "batch（使用 model.generate_batch()），"
                                      "chat（使用 model.generate_chat()），"
                                      "generate（使用 model.generate()，仅部分模型支持）。"
                                      "如果不指定，会根据模型路径或模型类型自动推断"))
    generation_group.add_argument("--dataset_generation_config_path", type=str, default=None,
                                help="数据集生成配置JSON文件路径，用于自定义各数据集的prompt和生成参数")

def get_model_args():
    """获取仅包含模型参数的解析器（用于测试）"""
    parser = argparse.ArgumentParser(description="模型参数配置")
    add_model_args(parser)
    return parser.parse_args()

if __name__ == "__main__":
    # 测试模型参数
    args = get_model_args()
    print("模型参数配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")