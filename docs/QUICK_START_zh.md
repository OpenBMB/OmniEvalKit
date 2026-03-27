# 快速上手指南

欢迎使用 Omni-Eval Kit (o_e_Kit)！本指南将帮助您快速配置并运行第一次评估。

## 第一步：环境配置

1.  **克隆项目** (如果您尚未操作)

    ```bash
    git clone https://github.com/OpenBMB/OmniEvalKit.git
    cd omnievalkit
    ```

2.  **安装依赖**

    **推荐方式：使用 [uv](https://docs.astral.sh/uv/)（安装速度快 10-100 倍）**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync --all-extras
    ```

    **使用 pip**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -e ".[all]"
    ```


## 第二步：准备数据集

评测数据集托管在 HuggingFace 上，使用下载脚本可一键获取：

```bash
# 下载全部数据集（约 108GB）
python scripts/hf_download.py --output_dir ./data

# 或只下载指定数据集（如 GigaSpeech）
python scripts/hf_download.py --datasets gigaspeech_test --output_dir ./data

# 查看可用数据集列表
python scripts/hf_download.py --list
```

下载完成后，`data/` 目录会自动生成框架所需的目录结构：

```
omnievalkit/
└── data/
    └── audio/
        └── asr/
            └── gigaspeech/
                ├── test.jsonl      # 标注文件
                └── test_files/     # 音频文件
```

详细说明请参阅 [数据集下载指南](./guides/DATA_DOWNLOAD_zh.md)。

## 第三步：运行评估

确保您在 `omnievalkit` 目录下，然后使用 `torchrun` 运行评估：

```bash
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/your/model \
    --pt_path /path/to/your/checkpoint.pt \
    --answer_path ./results \
    --model_name my_eval \
    --batchsize 4 \
    --max_sample_num 10 \
    --eval_gigaspeech_test
```

## 第四步：查看结果

评估完成后，结果会保存在 `--answer_path` 参数指定的目录中（默认为 `./results/` 或 `./answers_batch_test/`）。

目录结构如下：
```
<answer_path>/
└── <模型名称>/
    ├── <时间戳>/
    │   └── gigaspeech_test.json  # 包含了逐条样本的详细预测结果
    └── result.json               # 最终聚合的评估分数 (例如: 整体WER)
```

至此，您已经成功完成了一次评估！如需了解更多关于架构设计和如何扩展本框架的信息，请参阅其他英文文档。 