# 命令行参数详解

本文档详细解释了在运行 `eval_main.py` 脚本时可以使用的所有命令行参数。参数管理采用模块化设计，分为三个主要类别。

## 目录

1. [模型参数](#1-模型参数)
2. [数据集参数](#2-数据集参数)
3. [运行时参数](#3-运行时参数)
4. [评估控制参数](#4-评估控制参数)
5. [示例用法](#5-示例用法)

---

## 1. 模型参数

模型相关参数定义在 `o_e_Kit/utils/args/model_args.py` 中。

### 基础配置
-   `--model_type <type>`
    -   **功能**: 选择要使用的模型类型
    -   **可选值**: `minicpmo`, `minicpmo_duplex_demo`, `whisper`, `qwen3_omni`, `gemini_omni`
    -   **默认值**: `minicpmo`

-   `--model_name <name>`
    -   **功能**: 模型标识名称，用于结果文件命名
    -   **默认值**: `minicpm26o`

-   `--generate_method <method>`
    -   **功能**: 选择模型的推理方法
    -   **可选值**: `batch`, `chat`, `generate`
    -   **默认值**: 根据模型类型自动推断

### 模型路径
-   `--model_path <path>`
    -   **功能**: 模型文件所在目录
    -   **默认值**: 根据模型类型自动设置

-   `--tokenizer_path <path>`
    -   **功能**: 分词器路径
    -   **默认值**: 与模型路径相同

-   `--pt_path <path>`
    -   **功能**: 预训练权重文件路径（.pt文件）
    -   **默认值**: `None`

-   `--config_path <path>`
    -   **功能**: 自定义配置文件路径
    -   **默认值**: `None`

-   `--dataset_generation_config_path <path>`
    -   **功能**: 数据集生成配置JSON文件路径，用于自定义各数据集的prompt和生成参数
    -   **默认值**: `None`

## 2. 数据集参数

数据集参数通过配置驱动方式管理，定义在 `o_e_Kit/utils/args/dataset_args.py` 中。

### 自动生成的参数

每个在 `DATASET_REGISTRY` 中注册的数据集会自动生成以下参数：

- `--eval_<dataset_name>`: 启用该数据集的评估（布尔标志）
- `--<dataset_name>_data_prefix_dir`: 数据文件目录路径
- `--<dataset_name>_annotation_path`: 标注文件路径

### 示例数据集参数

**GigaSpeech测试集:**
- `--eval_gigaspeech_test`: 启用GigaSpeech测试集评估
- `--gigaspeech_test_data_prefix_dir`: 音频文件目录
- `--gigaspeech_test_annotation_path`: 标注文件路径

**AudioQA1M数据集:**
- `--eval_audioqa1m`: 启用AudioQA1M评估
- `--audioqa1m_data_prefix_dir`: 数据目录
- `--audioqa1m_annotation_path`: 标注文件

### 批量评估控制
- `--eval_all`: 启用所有数据集评估
- `--eval_all_audio`: 启用所有音频数据集
- `--eval_all_video`: 启用所有视频数据集
- `--eval_all_omni`: 启用所有多模态数据集

## 3. 运行时参数

运行时参数定义在 `o_e_Kit/utils/args/runtime_args.py` 中。

### 批处理和采样
- `--batchsize <size>`: 批处理大小（默认: 2）
- `--max_sample_num <num>`: 最大样本数，用于快速测试（默认: None）
- `--max_seq_len <len>`: 最大序列长度（默认: 8192）

### 输出设置
- `--answer_path <path>`: 结果保存路径（默认: `./results/`）
- `--save_interval <num>`: 保存间隔（默认: 1000）

### 分布式设置
- `--local_rank`: 分布式训练的本地进程rank
- `--world_size`: 总进程数
- `--master_port`: 主节点端口

### 设备配置
- `--device`: 运行设备（cuda/cpu）
- `--dtype`: 数据类型（fp16/bf16/fp32）

## 4. 评估控制参数

### 评估模式
- `--eval_mode`: 评估模式（single/batch/streaming）
- `--eval_metrics`: 使用的评估指标列表

### 特殊配置
- `--streaming_context_time`: StreamingBench上下文时间（秒）
- `--streaming_tasks`: StreamingBench任务列表
- `--livecc_data_type`: LiveCC数据类型（clipped/frames）

## 5. 示例用法

### 基础评估
```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/model \
    --eval_gigaspeech_test \
    --batchsize 4
```

### 覆盖数据集路径
```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_audioqa1m \
    --audioqa1m_data_prefix_dir /custom/path/to/data/ \
    --audioqa1m_annotation_path /custom/path/to/ann.jsonl
```

### 批量评估所有音频数据集
```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_all_audio \
    --max_sample_num 100 \
    --answer_path ./test_results/
```

## 查看所有可用参数

要查看完整的参数列表和当前值：

```bash
python eval_main.py --help
```

## 配置文件支持

除了命令行参数，你还可以通过配置文件管理参数：

1. **模型配置**: 通过 `--config_path` 指定JSON格式的模型配置
2. **数据集路径**: 可以通过环境变量覆盖默认路径
3. **批量配置**: 可以创建shell脚本预设常用参数组合

## 参数优先级

参数值的优先级从高到低：
1. 命令行显式指定的参数
2. 环境变量设置的值
3. 配置文件中的值
4. 代码中的默认值
