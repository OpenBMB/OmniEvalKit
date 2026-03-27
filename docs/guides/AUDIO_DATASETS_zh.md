# 音频数据集适配指南

## 概述

本框架支持多种主流音频ASR数据集的加载和处理。通过统一的 `AudioEvalDataset` 类，可以自动识别和处理不同格式的标注文件。

## 已支持的数据集

完整的数据集列表请参阅 HuggingFace 数据集仓库：**[OmniEvalKit/omnievalkit-dataset](https://huggingface.co/datasets/OmniEvalKit/omnievalkit-dataset)**

涵盖 ASR（语音识别）、QA（问答）、多任务音频理解（MMAU、MMSU、MMAR）、Caption（音频描述）、Classification（音频分类）、Emotion（情感识别）等多种任务类型。

## 标注文件格式要求

框架统一使用 **JSONL格式**（JSON Lines，每行一个JSON对象），简化数据处理流程。

### ASR任务必需字段
- **WavPath** - 音频文件路径（相对于 data_prefix_dir）
- **text** 或 **sentence** - 转录文本

### QA任务必需字段
- **WavPath** - 音频文件路径（相对于 data_prefix_dir）
- **question** - 问题文本
- **answer** - 答案文本
- **choices** - 选项列表（可选，用于多选题）

### 示例格式

#### ASR任务格式
```json
{"WavPath": "test/S0770/BAC009S0770W0259.wav", "text": "这是一段中文语音"}
{"WavPath": "common_voice_en_123.mp3", "sentence": "This is English speech"}
```

#### QA任务格式
```json
{"WavPath": "audio/test_001.wav", "question": "What sound is this?", "answer": "Dog barking", "choices": ["Dog barking", "Cat meowing", "Bird chirping", "Car horn"]}
```

#### Caption任务格式
```json
{"WavPath": "audio/test_001.wav", "caption": "A dog is barking loudly in the background"}
```

## 使用方法

### 1. 下载数据集

```bash
# 从 HuggingFace 一键下载并还原数据
python scripts/hf_download.py --output_dir ./data

# 详见 docs/guides/DATA_DOWNLOAD_zh.md
```

### 2. 加载数据集

```python
from o_e_Kit.datasets.audio_datasets import AudioEvalDataset

# 加载GigaSpeech数据集
dataset = AudioEvalDataset(
    annotation_path='./data/gigaspeech/test.jsonl',
    data_prefix_dir='./data/gigaspeech/test_files/',
    dataset_name='gigaspeech_test'
)

# 获取样本
idx, paths, annotation = dataset[0]
print(f"音频路径: {paths['audio_path']}")
print(f"转录文本: {annotation['gt_answer']}")
```

### 3. 在评估框架中使用

数据集已在 `o_e_Kit/utils/args/dataset_args.py` 中注册，可以通过命令行参数启用：

```bash
# 评估ASR数据集
python eval_main.py --eval_gigaspeech_test --eval_wenetspeech_test_net

# 评估多任务音频理解数据集
python eval_main.py --eval_mmau_test_mini --eval_mmsu_bench --eval_mmar_bench

# 评估所有音频数据集
python eval_main.py --eval_all_audio
```

## 添加新数据集

### 步骤1：准备JSONL格式的标注文件

确保你的标注文件是JSONL格式，每行包含：
- **WavPath** - 音频文件路径
- **text** 或 **sentence** - 转录文本

如果原始数据不是JSONL格式，需要先转换。

### 步骤2：放置数据文件

将数据文件放到 `data/` 对应子目录下：

```bash
mkdir -p data/audio/asr/my_dataset
cp /path/to/your/audio_files data/audio/asr/my_dataset/test_files/
cp /path/to/your/annotation.jsonl data/audio/asr/my_dataset/test.jsonl
```

### 步骤3：在dataset_args.py中注册

```python
DatasetConfig(
    name="new_dataset_test",
    display_name="New Dataset Test",
    category="audio",
    subcategory="asr",  # 或 "qa", "caption", "cls", "emotion" 等
    paths={
        "data_prefix_dir": "./data/new_dataset/test/",
        "annotation_path": "./data/new_dataset/test.jsonl"
    },
    description="新数据集描述"
),
```

### 步骤4：在audio_datasets.py中添加测试配置（可选）

在 `o_e_Kit/datasets/audio_datasets.py` 的 `test_configs` 列表中添加测试配置：

```python
{
    'name': 'New Dataset Test',
    'annotation_path': './data/new_dataset/test.jsonl',
    'data_prefix_dir': './data/new_dataset/test/',
    'dataset_name': 'new_dataset_test'
},
```

## 数据集格式示例

### LibriSpeech (.txt格式)
```
103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED
103-1240-0001 MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD
```

### CommonVoice (.tsv格式)
```
client_id	path	sentence	up_votes	down_votes	age	gender	accent	locale	segment
client123	common_voice_en_123.mp3	Hello world	2	0	twenties	male	us	en	train
```

### AISHELL (.txt格式)
```
BAC009S0002W0122	而 对 楼市 成交 抑制 作用 最 大 的 限 购
BAC009S0002W0123	也 成为 地方 政府 的 眼中 钉
```

## 常见问题

### Q: 数据集加载失败，提示找不到标注文件
A: 检查数据是否已下载，运行 `ls -la ./data/` 确认文件存在。如需重新下载，执行 `python scripts/hf_download.py --output_dir ./data`

### Q: 音频文件路径不正确
A: 检查 `data_prefix_dir` 参数是否正确，确保路径拼接后能找到音频文件

### Q: 标注格式不被识别
A: 查看错误信息，检查字段名是否在支持的列表中，或者添加新的字段映射

## 已完成的功能

- [x] 支持主流ASR数据集（中英文、多语言）
- [x] 支持QA问答任务数据集（VoiceBench系列、AudioQA1M等）
- [x] 支持多任务音频理解数据集（MMAU、MMSU、MMAR）
- [x] 支持音频字幕数据集（AudioCaps、ClothoCaption、WavCaps）
- [x] 支持音频分类数据集（VocalSound）
- [x] 支持情感识别数据集（MELD）
- [x] 统一的JSONL格式处理
- [x] 自动过滤超长音频（>30秒）
- [x] 支持duration字段优化加载速度

## 下一步计划

- [ ] 添加更多数据集的实际测试
- [ ] 支持更多音频理解任务数据集（ESC-50、AudioSet等）
- [ ] 添加数据集统计和可视化功能
- [ ] 支持数据集的自动下载和预处理
- [ ] 支持流式音频处理
- [ ] 添加数据增强功能
