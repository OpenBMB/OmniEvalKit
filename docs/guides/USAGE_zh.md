# 使用指南

本指南将解释如何使用我们提供的脚本来运行评估，以及如何解读评估结果。

## 1. 运行评估

使用 `torchrun` 调用 `eval_main.py` 来运行评估。

### 快速开始

1.  **进入项目根目录**:

    ```bash
    cd /path/to/your/project/omnievalkit
    ```

2.  **运行评估**:

    ```bash
    torchrun --nproc_per_node=1 eval_main.py \
        --model_path /path/to/your/model \
        --pt_path /path/to/your/checkpoint.pt \
        --answer_path "./results/" \
        --model_name "my_model" \
        --model_type "minicpmo" \
        --max_sample_num 10 \
        --batchsize 2 \
        --eval_gigaspeech_test
    ```

### 多 GPU 评估

通过 `--nproc_per_node` 控制使用的 GPU 数量：

```bash
torchrun --nproc_per_node=4 --master_port=29500 eval_main.py \
    --model_path /path/to/your/model \
    --model_type minicpmo \
    --pt_path /path/to/your/checkpoint.pt \
    --answer_path ./results \
    --model_name my_model \
    --batchsize 4 \
    --eval_gigaspeech_test
```

要查看所有可用的参数及其说明，请运行：

```bash
python eval_main.py --help
```

## 2. 理解输出结果

评估运行完成后，结果将保存在由 `--answer_path` 参数指定的目录中。其目录结构如下：

```
<answer_path>/
└── <模型名称>/
    ├── <时间戳>/
    │   └── <数据集名称>.json       # 原始预测结果和逐条样本的详细信息
    └── result.json                   # 最终的、聚合后的评估分数
```

-   **`<数据集名称>.json`**: 该文件包含了一个列表，其中每个元素都是一条样本的详细预测信息，包括标准答案、模型输出以及任何逐样本的分数（如WER的详细错误）。
-   **`result.json`**: 该文件包含了整个数据集的最终宏观评估指标（例如，总体的 WER 分数）。

您可以查看这些文件来详细分析您模型的性能。
