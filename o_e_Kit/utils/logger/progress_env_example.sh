#!/bin/bash
# 进度显示环境配置示例
# 使用方法: source progress_env_example.sh

echo "🔧 设置进度显示环境配置..."

# 基础配置
export PROGRESS_LOG_INTERVAL=5          # 每5个item输出一次日志
export PROGRESS_FORCE_CLOUD=false       # 是否强制使用云端模式
export PROGRESS_ENABLE_EMOJIS=true      # 是否启用emoji
export PROGRESS_LOG_FORMAT="%(asctime)s - %(levelname)s - %(message)s"  # 日志格式

echo "✅ 进度显示配置已设置:"
echo "  PROGRESS_LOG_INTERVAL=$PROGRESS_LOG_INTERVAL"
echo "  PROGRESS_FORCE_CLOUD=$PROGRESS_FORCE_CLOUD" 
echo "  PROGRESS_ENABLE_EMOJIS=$PROGRESS_ENABLE_EMOJIS"
echo ""

# 使用示例
echo "📋 使用示例:"
echo "# Python代码中现在可以简单使用："
echo "from o_e_Kit.utils.simple_progress import smart_progress"
echo ""
echo "for batch in smart_progress(dataloader, desc='训练模型'):"
echo "    # 训练逻辑，不需要指定log_interval了"
echo "    pass"
echo ""

# 不同环境的配置示例
echo "🌍 不同环境的配置建议:"
echo ""
echo "# 开发环境 (详细日志)"
echo "export PROGRESS_LOG_INTERVAL=3"
echo "export PROGRESS_ENABLE_EMOJIS=true"
echo ""
echo "# 生产环境 (简洁日志)"
echo "export PROGRESS_LOG_INTERVAL=20"
echo "export PROGRESS_ENABLE_EMOJIS=false"
echo "export PROGRESS_FORCE_CLOUD=true"
echo ""
echo "# 集群/云端环境 (云端友好)"
echo "export PROGRESS_LOG_INTERVAL=50"
echo "export PROGRESS_ENABLE_EMOJIS=false"
echo "export PROGRESS_FORCE_CLOUD=true"
