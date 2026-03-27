# 进度显示工具使用说明

## 🎯 解决的问题

`tqdm` 在云端监控和日志文件中存在以下问题：
- ❌ 产生大量ANSI控制字符，污染日志文件  
- ❌ 动态更新导致日志难以解析
- ❌ `log_interval` 参数不兼容，导致错误
- ❌ 不适合非交互式环境（集群、容器、CI/CD）

## ✅ 解决方案

### 核心特性
- 🔧 **环境配置驱动**: 通过环境变量统一配置，无需在代码中重复指定
- 🚀 **智能环境检测**: 自动选择最适合的进度显示方式
- 📊 **云端友好日志**: 结构化输出，易于解析和监控
- 🎛️ **灵活配置**: 支持emoji开关、日志格式自定义等

## 🚀 快速开始

### 1. 环境配置（推荐）

```bash
# 设置环境变量
export PROGRESS_LOG_INTERVAL=5        # 每5个item输出一次
export PROGRESS_ENABLE_EMOJIS=true    # 启用emoji
export PROGRESS_FORCE_CLOUD=false     # 不强制云端模式
```

### 2. 代码使用

```python
from o_e_Kit.utils.simple_progress import smart_progress

# 简单使用（使用环境配置）
for batch in smart_progress(dataloader, desc="训练模型"):
    # 训练逻辑
    pass

# 临时覆盖配置
for batch in smart_progress(dataloader, desc="特殊任务", log_interval=2):
    # 处理逻辑
    pass
```

### 3. 替换现有 tqdm 代码

```python
# 原来
from tqdm import tqdm
for batch in tqdm(dataloader, desc="Running inference", log_interval=5):  # ❌ 会报错
    pass

# 现在
from o_e_Kit.utils.simple_progress import smart_progress
for batch in smart_progress(dataloader, desc="Running inference"):  # ✅ 使用环境配置
    pass
```

## 🔧 环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `PROGRESS_LOG_INTERVAL` | `10` | 日志输出间隔 |
| `PROGRESS_FORCE_CLOUD` | `false` | 强制使用云端模式 |
| `PROGRESS_ENABLE_EMOJIS` | `true` | 是否启用emoji |
| `PROGRESS_LOG_FORMAT` | `%(asctime)s - %(levelname)s - %(message)s` | 日志格式 |

### 不同环境的推荐配置

#### 开发环境（详细日志）
```bash
export PROGRESS_LOG_INTERVAL=3
export PROGRESS_ENABLE_EMOJIS=true
```

#### 生产环境（简洁日志）
```bash
export PROGRESS_LOG_INTERVAL=20
export PROGRESS_ENABLE_EMOJIS=false
export PROGRESS_FORCE_CLOUD=true
```

#### 集群/云端环境
```bash
export PROGRESS_LOG_INTERVAL=50
export PROGRESS_ENABLE_EMOJIS=false
export PROGRESS_FORCE_CLOUD=true
```

## 📊 输出效果对比

### tqdm输出（问题）
```
Training: 42%|████▎     | 423/1000 [01:23<02:01,  4.74it/s]
Training: 43%|████▍     | 434/1000 [01:25<02:00,  4.72it/s]
```

### 云端友好输出（解决方案）
```
2024-07-28 18:45:01 - INFO - 🚀 开始 训练模型: 总共 1000 个项目
2024-07-28 18:45:03 - INFO - 📊 训练模型 进度: 100/1000 (10.0%) 速度: 4.76it/s, 预计剩余: 189s
2024-07-28 18:45:05 - INFO - 📊 训练模型 进度: 200/1000 (20.0%) 速度: 4.74it/s, 预计剩余: 169s
2024-07-28 18:45:07 - INFO - ✅ 训练模型 完成: 1000/1000 (100.0%) 耗时: 210.5s, 平均速度: 4.75it/s
```

## 🛠️ 运行时配置

```python
from o_e_Kit.utils.simple_progress import set_progress_config, get_progress_config

# 查看当前配置
config = get_progress_config()
print(config)

# 修改配置
set_progress_config(
    log_interval=5,
    enable_emojis=False,
    force_cloud=True
)

# 配置会影响后续的 smart_progress 调用
for item in smart_progress(data, desc="使用新配置"):
    pass
```

## 🎛️ 高级用法

### 1. 兼容 tqdm 接口

```python
from o_e_Kit.utils.simple_progress import tqdm_compatible as tqdm

# 可以直接替换 tqdm
for batch in tqdm(dataloader, desc="兼容模式"):
    pass
```

### 2. 多种别名

```python
from o_e_Kit.utils.simple_progress import (
    smart_progress,     # 主要接口
    progress_bar,       # 别名
    tqdm_compatible     # tqdm兼容
)
```

### 3. 环境检测

系统会自动检测运行环境：
- **交互式终端**: 使用传统 tqdm（如果可用）
- **云端/集群环境**: 自动使用云端友好模式
- **检测标识**: `SLURM_JOB_ID`, `PBS_JOBID`, `KUBERNETES_SERVICE_HOST`, `DOCKER_CONTAINER` 等

## 🔍 监控集成

云端友好的日志格式便于集成到监控系统：

```bash
# 提取进度信息
grep "📊.*进度" training.log | tail -1

# 统计完成任务
grep "✅.*完成" training.log | wc -l

# 监控处理速度
grep "速度:" training.log | sed 's/.*速度: \([0-9.]*\).*/\1/'
```

## 🚀 完整示例

```bash
# 1. 设置环境
source o_e_Kit/utils/progress_env_example.sh

# 2. 运行代码
python << EOF
from o_e_Kit.utils.simple_progress import smart_progress
import time

# 使用环境配置，代码简洁
for i in smart_progress(range(20), desc="处理数据"):
    time.sleep(0.1)
EOF
```

## 📝 最佳实践

1. **使用环境变量**：统一配置，避免在代码中硬编码
2. **合理设置间隔**：
   - 快速任务: `PROGRESS_LOG_INTERVAL=3-5`
   - 中等任务: `PROGRESS_LOG_INTERVAL=10-20`  
   - 大型任务: `PROGRESS_LOG_INTERVAL=50-100`
3. **云端环境**：关闭emoji，增加间隔
4. **监控系统**：解析结构化日志输出

这个解决方案完美适合自动化监控系统，提供清晰、可解析、可配置的进度信息！ 