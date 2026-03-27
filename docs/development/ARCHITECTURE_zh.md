# 架构与开发者指南

本文档为开发者提供一份关于 Omni-Eval Kit 架构的深度概览。它将解释项目的设计原则、代码库结构，以及如何扩展此框架以支持新的模型、数据集和评估指标。

## 1. 项目哲学

本工具包的设计遵循三大核心原则：

-   **模块化 (Modularity)**: 每个关键组件（模型处理、数据加载、推理、评估）都被解耦到独立的模块中。这使得整个系统更易于理解、调试和扩展。
-   **可扩展性 (Extensibility)**: 添加新功能应该是一个直接的过程，无需对现有代码进行大规模重构。我们为常见的用例提供了清晰的扩展点。
-   **清晰性优于复杂性 (Clarity over Complexity)**: 我们设计的核心工作流是明确且易于遵循的。例如，主要的评估逻辑都位于 `eval_main.py` 中，为整个流程提供了一个清晰的入口。

## 2. 代码库结构

本项目采用了标准的 Python 包结构，所有核心代码都组织在 `o_e_Kit` 包目录下：

```
omnievalkit/
├── eval_main.py            # 所有评估任务的主入口点
├── setup.py                # 包安装配置
├── data/                   # 评测数据集（通过 scripts/hf_download.py 下载）
├── docs/                   # 项目文档
├── scripts/                # 辅助脚本（数据集下载、格式转换）
├── results/                # 评估结果存储目录
├── configs/                # 全局配置文件
└── o_e_Kit/                # 核心包目录
    ├── __init__.py
    ├── datasets/           # 数据集实现
    │   ├── base_dataset.py         # 基础数据集类
    │   ├── audio_datasets.py       # 音频数据集
    │   ├── omni_datasets.py        # 多模态数据集
    │   └── duplex/                 # 双工相关辅助模块
    ├── models/             # 模型封装
    │   ├── __init__.py
    │   ├── minicpm/               # MiniCPM-O 系列模型
    │   │   ├── minicpmo.py        # MiniCPM-O 统一评测模型
    │   │   └── demo/              # Demo 脚本（Duplex、TTS 等）
    │   ├── qwen/                  # Qwen3-Omni 系列
    │   ├── gemini/                # Gemini API 评测
    │   └── asr/                   # ASR 基线模型（Whisper）
    ├── utils/              # 工具模块
    │   ├── args/                  # 参数管理（模块化设计）
    │   │   ├── dataset_args.py   # 数据集参数
    │   │   ├── model_args.py     # 模型参数
    │   │   └── runtime_args.py   # 运行时参数
    │   ├── metrics/               # 评估指标
    │   │   ├── evaluator_base.py # 基础评估器
    │   │   ├── wer_eval.py       # WER/CER 评估
    │   │   ├── evaluator_mqa.py  # 多模态问答评估
    │   │   └── ...
    │   ├── logger/                # 日志和进度追踪
    │   ├── text_normalization/    # 文本标准化
    │   ├── visualizer/            # 可视化报告
    │   ├── infer.py              # 分布式推理循环
    │   ├── dataloader.py         # 数据加载工具
    │   ├── dataset_loader.py     # 数据集加载器
    │   ├── model_loader.py       # 模型加载器
    │   └── evaluation_runner.py  # 评估运行器
    └── configs/            # 数据集生成配置
```

### 核心模块说明

-   **`eval_main.py`**: 主入口点，负责协调整个评估流程。它解析参数、初始化环境、加载模型和数据集、执行推理并触发评估。

-   **`o_e_Kit/datasets/`**: 采用面向对象设计，所有数据集都继承自 `BaseDataset`，按数据类型分类实现：
    -   `audio_datasets.py`: ASR、音频问答等音频任务数据集
    -   `video_datasets.py`: 视频理解、视频问答等视频任务数据集
    -   `omni_datasets.py`: 多模态综合任务数据集
    -   `duplex_datasets.py`: 支持双工交互的数据集

-   **`o_e_Kit/models/`**: 模型封装层，每个模型类可以实现多种生成方法（如 `generate_batch`, `generate_duplex`）

-   **`o_e_Kit/utils/args/`**: 参数管理采用模块化设计，将不同类型的参数分离：
    -   `dataset_args.py`: 使用配置驱动方式管理所有数据集参数
    -   `model_args.py`: 模型相关参数
    -   `runtime_args.py`: 运行时参数（分布式、设备等）

-   **`o_e_Kit/utils/metrics/`**: 评估指标模块，所有评估器继承自 `evaluator_base.py`，提供统一接口

-   **`o_e_Kit/utils/`**: 核心工具集
    -   `infer.py`: 管理推理循环和批处理
    -   `dataloader.py`: 提供统一的数据加载接口
    -   `dataset_loader.py`: 动态加载数据集
    -   `model_loader.py`: 动态加载模型
    -   `evaluation_runner.py`: 执行评估流程

## 3. 如何扩展本框架

这是本工具包最强大的地方。以下是如何添加新组件的指南。

### 如何添加一个新的数据集

1.  **在配置注册表中添加数据集**: 编辑 `o_e_Kit/utils/args/dataset_args.py`，在 `DATASET_REGISTRY` 列表中添加新的 `DatasetConfig`:
    ```python
    DatasetConfig(
        name="my_new_dataset",
        display_name="My New Dataset",
        category="audio",  # 或 "video", "omni", "duplex_audio" 等
        subcategory="asr",  # 可选：如 "qa", "caption" 等
        paths={
            "data_prefix_dir": "./data/my_new_dataset/",
            "annotation_path": "./data/my_new_dataset/test.jsonl"
        },
        default_enabled=False,
        description="我的新数据集描述"
    )
    ```

2.  **创建数据集类**: 根据数据类型，在相应文件中添加数据集类：
    -   音频数据集：在 `o_e_Kit/datasets/audio_datasets.py` 中添加
    -   视频数据集：在 `o_e_Kit/datasets/video_datasets.py` 中添加
    -   多模态数据集：在 `o_e_Kit/datasets/omni_datasets.py` 中添加
    
    数据集类应继承自 `BaseDataset` 并实现必要的方法：
    ```python
    class MyNewDataset(BaseDataset):
        def __init__(self, data_prefix_dir, annotation_path, **kwargs):
            super().__init__()
            # 加载数据
        
        def __getitem__(self, idx):
            # 返回统一格式的字典
            return {
                "wav_path": ...,
                "question": ...,
                "gt_answers": ...,
                # 其他必要字段
            }
    ```

3.  **更新数据集加载器**: 在 `o_e_Kit/utils/dataset_loader.py` 中添加新数据集的加载逻辑。

4.  **参数会自动生成**: 由于使用了配置驱动的方式，相关的命令行参数（如 `--eval_my_new_dataset`、`--my_new_dataset_data_prefix_dir` 等）会自动生成。

### 如何添加一个新的模型

1.  **在模型参数中注册**: 编辑 `o_e_Kit/utils/args/model_args.py`，添加新模型的参数配置。

2.  **创建模型封装类**: 在 `o_e_Kit/models/` 目录下创建新的模型目录和文件：
    ```python
    # o_e_Kit/models/my_model/my_model.py
    class MyModel:
        def __init__(self, model_path, device, **kwargs):
            # 加载模型权重
            self.model = load_model(model_path)
            self.device = device
        
        def generate_batch(self, **batch):
            # 批量生成方法
            return ["response1", "response2", ...]
        
        def generate_duplex(self, **batch):
            # 双工生成方法（可选）
            return ["response1", "response2", ...]
    ```

3.  **更新模型加载器**: 在 `o_e_Kit/utils/model_loader.py` 中添加新模型的加载逻辑：
    ```python
    elif args.model_type == "my_model":
        from o_e_Kit.models.my_model import MyModel
        return MyModel(args.model_path, args.device, **kwargs)
    ```

4.  **注册生成方法**: 在 `o_e_Kit/utils/infer.py` 中添加对新生成方法的支持。

### 如何添加一个新的评估指标

1.  **创建评估器类**: 在 `o_e_Kit/utils/metrics/` 目录下创建新的评估器（例如 `evaluator_my_metric.py`）：
    ```python
    from .evaluator_base import EvaluatorBase
    
    class MyMetricEvaluator(EvaluatorBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def evaluate(self, predictions, references):
            # 计算评估指标
            scores = []
            for pred, ref in zip(predictions, references):
                score = self.compute_score(pred, ref)
                scores.append(score)
            return scores
        
        def summary(self):
            # 返回格式化的报告和最终分数
            avg_score = sum(self.scores) / len(self.scores)
            report = f"Average Score: {avg_score:.4f}"
            return report, avg_score
    ```

2.  **注册评估器**: 在 `o_e_Kit/utils/evaluation_runner.py` 中添加新评估器的调用逻辑。

3.  **配置数据集使用新指标**: 在数据集配置或运行参数中指定使用新的评估指标。

## 4. 最佳实践

### 配置驱动开发
项目采用配置驱动的方式管理数据集，所有数据集信息集中在 `DATASET_REGISTRY` 中，便于维护和扩展。

### 模块化设计
- 参数管理分离：数据集参数、模型参数、运行时参数独立管理
- 评估器继承体系：所有评估器继承自基类，确保接口一致性
- 数据集分类管理：按数据类型组织数据集实现

### 代码复用
- 使用 `BaseDataset` 作为所有数据集的基类
- 通用工具函数集中在 `utils/` 目录下

## 5. 整体运行流程

### 执行流程概览

```
用户命令 → eval_main.py → 参数解析 → 环境初始化 → 模型加载 → 数据集加载 → 推理执行 → 评估计算 → 结果输出
run_all_evaluations 
->evaluate_all_audio_datasets 
    ->infer_and_evaluate 
        ->run_inference
        得到一个数据集的预测结果
    ->evaluate_dataset
        得到一个数据集的评估结果
->save_evaluation_results
保存所有结果
```

### 详细流程说明

#### 1. 启动阶段
- 用户通过 `torchrun` 命令启动评估
- 支持单机单卡、单机多卡、多机多卡的分布式运行

#### 2. 参数解析（`get_args.py`）
- 解析命令行参数，包括：
  - 模型参数（`model_args.py`）：模型路径、类型、配置等
  - 数据集参数（`dataset_args.py`）：数据集路径、评估标志等
  - 运行时参数（`runtime_args.py`）：批大小、设备、分布式设置等

#### 3. 环境初始化（`eval_main.py`）
- 初始化分布式环境（如果需要）
- 设置随机种子
- 创建输出目录
- 配置日志系统

#### 4. 模型加载（`model_loader.py`）
- 根据 `--model_type` 参数动态加载对应的模型类
- 初始化模型并加载权重
- 将模型移至指定设备（GPU/CPU）

#### 5. 数据集加载（`dataset_loader.py`）
- 根据 `--eval_*` 标志确定要评估的数据集
- 动态加载对应的数据集类
- 创建 DataLoader 用于批量处理

#### 6. 推理执行（`infer.py`）
- 遍历 DataLoader 中的批次
- 调用模型的生成方法（如 `generate_batch`）
- 收集模型输出
- 支持分布式推理同步

#### 7. 评估计算（`evaluation_runner.py` + `metrics/`）
- 根据数据集类型选择对应的评估器
- 计算评估指标（WER、BLEU、准确率等）
- 生成详细的评估报告

#### 8. 结果输出
- 保存原始预测结果到 `<answer_path>/<model_name>/<timestamp>/<dataset>.json`
- 保存汇总结果到 `<answer_path>/<model_name>/result.json`
- 在控制台打印评估摘要

### 关键执行路径

```python
# eval_main.py 中的主要流程
def main():
    # 1. 解析参数
    args = get_args()
    
    # 2. 初始化环境
    init_distributed_mode(args)
    
    # 3. 加载模型
    model = load_model(args)
    
    # 4. 对每个启用的数据集
    for dataset_name in get_enabled_datasets(args):
        # 5. 加载数据集
        dataset = load_dataset(args, dataset_name)
        
        # 6. 执行推理
        predictions = run_inference(model, dataset, args)
        
        # 7. 运行评估
        results = run_evaluation(predictions, dataset, args)
        
        # 8. 保存结果
        save_results(results, args)
```