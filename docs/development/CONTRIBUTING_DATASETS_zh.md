# 如何贡献一个新数据集

本指南详细说明了如何为 Omni-Eval Kit 框架添加一个新的评测数据集。

## 设计原则

- **格式统一**：所有数据集需要转换为框架支持的标准格式（JSONL）
- **配置驱动**：通过 `DATASET_REGISTRY` 统一管理数据集配置
- **代码复用**：优先复用现有的 `BaseDataset` 子类，避免重复开发
- **评测集成**：每个数据集需要有对应的评测方式

## 完整流程概览

1. **数据格式转换** → 2. **放置数据文件** → 3. **配置注册** → 4. **选择/实现数据集类** → 5. **配置评测方式** → 6. **测试验证**

## 步骤详解

### 1. 数据格式转换

需要将数据集转换为 **JSONL** 格式（每行一个JSON对象）。框架会自动适配不同的字段名。

#### 音频数据集格式要求

**必需字段：**
- `WavPath`: 音频文件的相对路径（相对于 data_prefix_dir）

**文本字段（至少包含一个）：**
- `text` / `sentence` / `transcription`: ASR转录文本
- `caption`: 音频描述任务
- `answer` / `answers`: 问答任务答案
- `reference`: 参考答案
- `label` / `normalized_text`: 其他文本标签

**可选字段：**
- `question` / `prompt`: 问题或提示词（用于QA任务）
- `id` / `audio_id`: 样本ID
- 其他自定义字段

#### 实际格式示例

```json
// ASR任务
{"WavPath": "audio/sample_001.wav", "text": "这是转录文本"}
{"WavPath": "audio/sample_002.wav", "sentence": "另一种文本字段名"}

// 问答任务
{"WavPath": "audio/qa_001.wav", "question": "音频中说了什么？", "answer": "正确答案"}
{"WavPath": "audio/qa_002.wav", "prompt": "请描述音频内容", "answers": "可能的答案"}

// 描述任务
{"WavPath": "audio/cap_001.wav", "caption": "鸟鸣声和流水声"}
```

#### 转换脚本示例

```python
# scripts/convert_dataset_to_jsonl.py
import json

def convert_to_jsonl(input_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in input_data:
            # 转换为框架格式
            formatted_item = {
                "WavPath": item["audio_file"],  # 必需：音频路径
                "text": item["transcription"],   # 文本内容
                "id": item.get("id", ""),        # 可选：ID
            }
            f.write(json.dumps(formatted_item, ensure_ascii=False) + '\n')

# 框架会自动识别并适配以下字段名：
# text, sentence, transcription, label, normalized_text, 
# answer, answers, caption, reference, question, prompt
```

### 2. 放置数据文件

将数据文件放到 `data/` 对应子目录下：

```bash
# 创建目录结构
mkdir -p data/audio/asr/my_dataset

# 放置数据文件
cp -r /path/to/original/audio/files data/audio/asr/my_dataset/audio_files
cp /path/to/converted/annotation.jsonl data/audio/asr/my_dataset/test.jsonl

# 验证
ls -la data/audio/asr/my_dataset/
```

目录结构示例：
```
data/
├── audio/
│   ├── asr/
│   │   ├── gigaspeech/
│   │   └── my_dataset/
│   │       ├── audio_files -> /path/to/audio/
│   │       └── test.jsonl -> /path/to/annotation.jsonl
│   └── qa/
├── video/
└── omni/
```

### 3. 在配置注册表中添加数据集

编辑 `o_e_Kit/utils/args/dataset_args.py`，在 `DATASET_REGISTRY` 中添加配置：

```python
DatasetConfig(
    name="my_dataset_asr",                    # 唯一标识符，用于命令行参数
    display_name="My Dataset ASR",            # 显示名称
    category="audio",                         # 主类别
    subcategory="asr",                        # 子类别：asr/qa/caption等
    paths={
        "data_prefix_dir": "./data/audio/asr/my_dataset/audio_files/",
        "annotation_path": "./data/audio/asr/my_dataset/test.jsonl"
    },
    default_enabled=False,                    # 是否默认启用
    description="My dataset for ASR evaluation"
)
```

这会自动生成命令行参数：
- `--eval_my_dataset_asr`: 启用评估
- `--my_dataset_asr_data_prefix_dir`: 数据目录
- `--my_dataset_asr_annotation_path`: 标注文件

### 4. 使用统一的数据集类

**音频数据集统一使用 `AudioEvalDataset` 类，无需创建新类！**

#### 4.1 AudioEvalDataset 的强大适配能力

`AudioEvalDataset` 会自动：
- 识别多种字段名变体（text, sentence, transcription, caption等）
- 推断任务类型（ASR, QA, Caption）
- 处理特定数据集的路径特殊性
- 跳过超过30秒的音频文件

#### 4.2 在数据集加载器中注册

编辑 `o_e_Kit/utils/dataset_loader.py`，添加数据集加载逻辑：

```python
def load_dataset(args, dataset_name):
    # 音频数据集统一使用 AudioEvalDataset
    if dataset_name in ["my_dataset_asr", "my_dataset_qa", "my_dataset_caption"]:
        from o_e_Kit.datasets.audio_datasets import AudioEvalDataset
        return AudioEvalDataset(
            annotation_path=getattr(args, f"{dataset_name}_annotation_path"),
            data_prefix_dir=getattr(args, f"{dataset_name}_data_prefix_dir"),
            dataset_name=dataset_name  # 传入数据集名称用于特殊处理
        )
```

#### 4.3 处理特殊路径需求

如果您的数据集有特殊的路径处理需求，可以在 `AudioEvalDataset.process_path` 方法中添加：

```python
def process_path(self, path):
    """处理特定数据集的路径转换"""
    # GigaSpeech 特殊处理
    if 'gigaspeech' in self.dataset_name.lower():
        path = path.replace('_metadata', '')
    
    # VoiceBench 特殊处理
    elif 'voicebench' in self.dataset_name.lower():
        if 'test/' in path:
            path = path.replace('/test/', '/')
    
    # 添加您的数据集特殊处理
    elif 'my_dataset' in self.dataset_name.lower():
        # 例如：移除特定前缀或转换路径格式
        path = path.replace('old_prefix/', '')
    
    return path
```

#### 4.4 视频和多模态数据集

- **视频数据集**：使用 `VideoEvalDataset`
- **多模态数据集**：使用 `OmniEvalDataset`  
- **双工数据集**：使用 `DuplexDataset`

这些基类同样提供了灵活的字段适配能力。

### 5. 配置评测方式

每个数据集需要指定对应的评测方法。

#### 5.1 在评测运行器中注册

编辑 `o_e_Kit/utils/evaluation_runner_audio.py`，添加数据集与评测器的映射：

```python
def evaluate_dataset(dataset_name, predictions, args):
    """根据数据集选择评测器"""
    
    # ASR类数据集使用WER评测
    if dataset_name in ["gigaspeech_test", "my_dataset_asr"]:
        from o_e_Kit.utils.metrics.wer_eval import WER_Eval
        evaluator = WER_Eval(
            lang='en' if 'gigaspeech' in dataset_name else 'zh',
            metric='wer'
        )
        
    # QA类数据集使用准确率评测
    elif dataset_name in ["audioqa1m", "my_dataset_qa"]:
        from o_e_Kit.utils.metrics.evaluator_openqa import OpenQAEvaluator
        evaluator = OpenQAEvaluator()
        
    # 其他评测器...
```

#### 5.2 选择合适的评测器

框架提供的评测器：
- `WER_Eval`: 语音识别（WER/CER）
- `OpenQAEvaluator`: 开放问答
- `CaptionEvaluator`: 描述生成（BLEU等）
- `MCQEvaluator`: 多选题
- `SafetyEvaluator`: 安全性评测

### 6. 测试验证

#### 6.1 小规模测试

```bash
# 测试数据加载
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --eval_my_dataset_asr \
    --max_sample_num 10 \
    --batchsize 2
```

#### 6.2 检查输出

确认以下内容：
1. 数据正确加载
2. 推理正常执行
3. 评测指标合理
4. 结果文件生成

## 常见问题

### Q1: 如何处理多种格式的标注？
A: 在数据格式转换阶段统一处理，或在 `__getitem__` 中添加兼容逻辑。

### Q2: 数据集太大怎么办？
A: 使用 `--datasets` 参数只下载指定数据集，支持流式加载，或创建子集用于测试。

### Q3: 需要自定义评测指标怎么办？
A: 参考[贡献评测方法指南](CONTRIBUTING_EVALS_zh.md)创建新的评测器。

## 完整示例：添加新的中文ASR数据集

假设您有一个中文语音识别数据集，以下是完整的集成步骤：

### 步骤1：准备数据格式

```python
# scripts/convert_my_chinese_asr.py
import json
import os

# 假设原始数据格式
original_data = [
    {"id": "001", "audio_file": "wav/001.wav", "transcript": "你好世界"},
    {"id": "002", "audio_file": "wav/002.wav", "transcript": "今天天气真好"},
]

# 转换为框架格式
with open('my_chinese_asr.jsonl', 'w', encoding='utf-8') as f:
    for item in original_data:
        jsonl_item = {
            "WavPath": item["audio_file"],      # 必需字段
            "text": item["transcript"],          # 框架会自动识别
            "id": item["id"]                     # 可选字段
        }
        f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')
```

### 步骤2：放置数据文件

```bash
# 创建目录
mkdir -p data/audio/asr/my_chinese_asr

# 放置音频文件和标注
cp -r /path/to/my_dataset/wav data/audio/asr/my_chinese_asr/wav
cp /path/to/my_chinese_asr.jsonl data/audio/asr/my_chinese_asr/test.jsonl

# 验证
ls -la data/audio/asr/my_chinese_asr/
```

### 步骤3：注册配置

```python
# 在 o_e_Kit/utils/args/dataset_args.py 的 DATASET_REGISTRY 中添加
DatasetConfig(
    name="my_chinese_asr",
    display_name="My Chinese ASR Dataset",
    category="audio",
    subcategory="asr",
    paths={
        "data_prefix_dir": "./data/audio/asr/my_chinese_asr/",
        "annotation_path": "./data/audio/asr/my_chinese_asr/test.jsonl"
    },
    default_enabled=False,
    description="自定义中文ASR数据集"
)
```

### 步骤4：注册数据集加载

```python
# 在 o_e_Kit/utils/dataset_loader.py 中添加
elif dataset_name == "my_chinese_asr":
    from o_e_Kit.datasets.audio_datasets import AudioEvalDataset
    return AudioEvalDataset(
        annotation_path=getattr(args, f"{dataset_name}_annotation_path"),
        data_prefix_dir=getattr(args, f"{dataset_name}_data_prefix_dir"),
        dataset_name=dataset_name
    )
```

### 步骤5：配置评测方法

```python
# 在 o_e_Kit/utils/evaluation_runner_audio.py 中添加
if dataset_name in ["wenetspeech_test_net", "aishell1_test", "my_chinese_asr"]:
    from o_e_Kit.utils.metrics.wer_eval import WER_Eval
    evaluator = WER_Eval(lang='zh', metric='cer')  # 中文使用CER
```

### 步骤6：运行评测

```bash
# 小规模测试
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/model \
    --eval_my_chinese_asr \
    --max_sample_num 10

# 完整评测
torchrun --nproc_per_node=1 eval_main.py \
    --model_type minicpmo \
    --model_path /path/to/model \
    --eval_my_chinese_asr \
    --batchsize 8
```

## 数据格式灵活性示例

`AudioEvalDataset` 支持多种字段名，以下都是有效的：

```json
// 这些都会被识别为文本内容
{"WavPath": "001.wav", "text": "转录文本"}
{"WavPath": "002.wav", "sentence": "另一种字段名"}
{"WavPath": "003.wav", "transcription": "也支持这个"}
{"WavPath": "004.wav", "label": "标签文本"}
{"WavPath": "005.wav", "normalized_text": "标准化文本"}

// 问答格式
{"WavPath": "qa1.wav", "question": "问题", "answer": "答案"}
{"WavPath": "qa2.wav", "prompt": "提示", "answers": "多个答案"}

// 描述格式
{"WavPath": "cap1.wav", "caption": "音频描述"}
``` 