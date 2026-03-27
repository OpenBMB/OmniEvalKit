# 评估框架模块说明

本目录包含评估框架的核心模块，主要负责模型加载、数据集加载和评估运行。

## 📁 模块结构

### 1. model_loader.py
**模型加载模块** - 负责加载各种类型的模型
- `load_model()`: 主函数，根据模型类型加载相应模型
- `load_model_hf()`: 加载HuggingFace格式的模型
- 支持的模型类型：
  - `minicpmo`: MiniCPM-O 统一模型（支持dataset_generation_config）
  - `minicpmo_duplex_demo`: MiniCPM-O Duplex Demo
  - `whisper`: Whisper ASR 基线模型
  - `qwen3_omni`: Qwen3-Omni 多模态理解模型
  - `gemini_omni`: Gemini 多模态 API 评测模型

### 2. dataset_loader.py
**数据集加载模块** - 显式处理每个数据集的加载
- `load_dataset()`: 统一的数据集加载接口
- `build_itembuilder_args()`: 构建duplex数据集参数
- 显式支持的数据集：
  - **音频ASR数据集**：GigaSpeech、WenetSpeech、LibriSpeech、CommonVoice、AISHELL-1、VoxPopuli、FLEURS、People's Speech、TED-LIUM
  - **音频QA数据集**：AudioQA1M、VoiceBench系列（9个子集）
  - **视频数据集**：OVOBench、StreamingBench
  - **多模态数据集**：VisionCap、OmniCap、LiveCC、AVEvent等

### 3. evaluation_runner.py
**主评估运行模块** - 协调所有评估任务
- `run_all_evaluations()`: 运行所有评估任务
- `evaluate_video_datasets()`: 评估视频数据集
- `evaluate_duplex_datasets()`: 评估Duplex数据集
- `evaluate_omni_datasets()`: 评估Omni数据集
- `save_evaluation_results()`: 保存评估结果

### 4. evaluation_runner_audio.py
**音频评估模块** - 显式处理每个音频数据集
- `evaluate_all_audio_datasets()`: 评估所有音频数据集
- `evaluate_voicebench_datasets()`: 评估VoiceBench系列
- 特点：每个数据集都有独立的评估代码块，便于调试和维护

## 🔧 使用方式

主程序（`eval_main.py`）的简洁流程：

```python
# 1. 初始化环境
torch.distributed.init_process_group(...)

# 2. 加载模型
model = load_model(args, device)

# 3. 运行所有评估
result = run_all_evaluations(args, model, device, time)

# 4. 保存结果
save_evaluation_results(result, args, time)
```

## 📊 评估流程

1. **音频数据集评估** → 分类处理ASR和QA任务
2. **视频数据集评估** → 处理OVOBench、StreamingBench等
3. **Duplex数据集评估** → 动态加载对应的duplex模型
4. **Omni数据集评估** → 处理全模态数据集

## 🚀 扩展指南

### 添加新模型
在`model_loader.py`的`load_model()`函数中添加新的模型类型判断。

### 添加新数据集
1. 在`dataset_loader.py`中显式添加数据集加载逻辑
2. 在相应的评估模块中添加评估代码块
3. 如果是音频数据集，在`evaluation_runner_audio.py`中添加

### 添加新的评估任务
1. 可以创建新的评估模块（如`evaluation_runner_xxx.py`）
2. 在`evaluation_runner.py`中导入并调用

## 🎯 设计理念

- **显式优于隐式**：每个数据集都有明确的处理代码
- **模块化设计**：功能分离，便于维护
- **清晰的输出**：每步都有明确的进度提示
- **易于调试**：可以轻松定位和修改特定数据集的处理逻辑
