# 数据集集成全流程指南

本文档说明如何将新的多模态评测数据集集成到 o_e_Kit 框架中。

## ⚠️ 必须修改的文件清单（复制此清单逐一检查）

集成新数据集 `xxx_dataset` 时，**必须**修改以下文件，漏掉任何一个都会报错：

```
□ 1. playground/convert_xxx.py              # 转换脚本（生成 JSONL）
□ 2. data/omni/raw_hf/xxx/                  # 数据目录（JSONL + 媒体文件）
□ 3. o_e_Kit/utils/args/dataset_args.py     # DatasetConfig 注册
□ 4. o_e_Kit/utils/dataset_loader.py        # load_dataset() 添加 elif 分支
□ 5. o_e_Kit/utils/evaluation_runner.py     # evaluate_omni_datasets() 添加评测逻辑
□ 6. o_e_Kit/utils/eval.py                  # ⚠️ 两处修改：
     □ 6a. dataset_name in [...] 列表中添加 "xxx_dataset"
     □ 6b. group_fields 字典中添加分组字段配置
□ 7. o_e_Kit/configs/ 下的所有生成配置文件（共6个）：
     □ omni_generation_configs.json
     □ omni_generation_configs_nosys.json
     □ omni_generation_configs_nosys_interleave.json      # ⚠️ 测试脚本默认用这个
     □ omni_generation_configs_nosys_interleave_sys.json
     □ omni_generation_configs_nosys_interleave_96.json
     □ omni_generation_configs_fullprompt.json
□ 8. 运行验证测试（见下方"验证集成"章节）
```

### 常见遗漏点 🔥

| 遗漏位置 | 报错信息 |
|---------|---------|
| `eval.py` 数据集列表 | `Unsupported dataset 'xxx'. Please add an evaluation flow...` |
| `configs/*.json` | `No config found for dataset 'xxx'` |
| `dataset_loader.py` | `Unsupported dataset: xxx` |
| `evaluation_runner.py` | 数据集不会被评测（静默跳过） |

## 概述

集成一个新数据集需要以下步骤：

1. **分析数据格式** - 了解原始数据结构
2. **转换脚本** - `playground/convert_xxx.py`
3. **提取音频** - 从视频提取 `.wav` 文件（⚠️ `load_av` 必需）
4. **放置数据文件** - `data/omni/raw_hf/`
5. **数据集配置** - `o_e_Kit/utils/args/dataset_args.py`
6. **数据集加载** - `o_e_Kit/utils/dataset_loader.py`
7. **评测运行** - `o_e_Kit/utils/evaluation_runner.py`
8. **生成配置** - `o_e_Kit/configs/` 下所有 JSON 配置文件（共6个）
9. **评测函数** - `o_e_Kit/utils/eval.py`（数据集列表 + 分组字段）
10. **验证测试** - 运行 eval_main.py 验证数据集加载和评测流程

## 快速检查清单（集成新 Benchmark 前先看一眼）

在真正写代码前，先把下面这些问题想清楚 / 勾一遍，可以大幅减少后面返工：

1. **任务与模态**
   - 这是 **MCQ / Open QA / Caption / 多任务混合** 中的哪一种？  
   - 输入模态是 **音频 / 视频 / 图片 / 多模态组合**？有几路媒体？（如多音频、多图、多视频）
   - 是否存在 **顺序/多轮问答**（例如 SQA）或 **流式 / 主动输出** 这类特殊交互形式？

2. **时间相关信息**
   - 数据里是否有 **时间戳 / 时间区间**（如 `time_stamp`, `time`, `realtime` 等）？  
   - 官方评测是否规定了 **固定上下文窗口**（例如「问题前 60 秒」）？  
   - 你打算用 **离线 chunk** 还是在线裁剪？（本仓库目前更推荐离线 chunk）

3. **路径与命名**
   - 原始标注中的视频 / 音频路径与真实磁盘上的文件 **一一对应吗**？有没有像 `*_1-25.mp4` vs `*.mp4` 这种不一致？  
   - 是否需要在转换脚本中做「**文件名规范化**」映射，或者先在数据准备阶段整理命名？

4. **音频来源与提取策略**
   - 这个 benchmark 需要音频吗？  
   - 音频是 **独立文件**（WAV/MP3）还是 **封装在视频里**？  
   - 准备在哪个阶段提取音频：
     - 转换时：`convert_xxx.py --extract-audio`（推荐，对 chunk 后片段提取）；
     - 还是在加载阶段由 `auto_extract_audio` 在线提取（仅适合小数据集或调试）。

5. **JSONL 统一 schema**
   - 是否已经设计好：每条样本里需要哪些字段？  
   - 必须字段：`VideoPath` / `WavPath` / `dataset_type` / `dataset_name` / `question` / `choices` / `gt_answer`。  
   - 还需要写入哪些元数据用于 **分组统计 / 分析**？例如：
     - `task` / `task_group`（OVOBench）  
     - `task_type` / `required_ability` / `video_categories`（StreamingBench）  
     - `sqa_context`（顺序问答的文本上下文）等。

6. **生成配置与 Prompt 设计**
   - 模型需要看到的 **prompt 模板** 是什么？是否随 benchmark 子任务变化？  
   - 是否需要额外占位符：`{media}` / `{sqa_context}` / `<audio_1>` / `<image_1>` 等？  
   - 预估每条样本需要的 **max_tokens / max_frames / max_fps** 是多少，是否需要降低 fps 以适应长视频？

7. **评测统计与分组**
   - 你希望最终报告按哪些字段做 **分组统计**？例如：
     - 按任务类型、题型、能力标签、视频类别、上层大类等。  
   - 这些字段是否已经在 JSONL 中写入，并在 `eval.py` 的 `group_fields` 中配置？

8. **运行前自检**
   - 转换脚本是否输出了 **missing_video / missing_clip** 等统计？这些数字是否为 0？  
   - 目录结构是否正确：`data_prefix_dir` + `VideoPath` 能否拼成真实文件路径？  
   - 用 `python3 eval_main.py --eval_xxx_dataset --max_sample_num 5` 跑几条，检查：
     - 媒体是否能正常加载；
     - 日志中是否有「未找到音频」之类 warning；
     - prompt 预览是否符合预期。

带着这份清单，再往下看「详细步骤」会更清晰。

## 详细步骤

### 步骤 1: 分析原始数据格式

首先分析新数据集的原始格式：

```bash
# 查看目录结构
ls -la /path/to/dataset/

# 查看 JSON/JSONL 文件结构
head -n 50 /path/to/dataset/data.json
python3 -c "import json; print(json.dumps(json.load(open('data.json'))[0], indent=2))"
```

需要了解：
- 视频/音频/图片路径格式
- 问题和答案格式
- 选项格式（MCQ）
- 元数据字段

### 步骤 2: 创建转换脚本（convert_xxx.py）

在 `playground/` 目录下创建转换脚本 `convert_xxx.py`：

```python
#!/usr/bin/env python3
"""
XXX 数据转换脚本
将原始数据转换为框架统一标准 JSONL 格式
"""

import os
import json
import argparse
from tqdm import tqdm


def convert_xxx_to_jsonl(input_path, output_path, data_root, verify_paths=True):
    """
    统一输出格式:
    {
        "VideoPath": "path/to/video.mp4",    # 视频路径
        "WavPath": "",                        # 音频路径（置空则自动提取）
        "ImagePath": "path/to/image.jpg",    # 图片路径（如果有）
        "dataset_type": "mcq",               # 任务类型: mcq, open_qa, caption
        "dataset_name": "xxx",               # 数据集名称
        "question": "问题文本",              # 问题
        "choices": ["选项A", "选项B", ...],  # MCQ 选项（不带字母前缀）
        "gt_answer": "A",                    # 答案（MCQ 用字母）
        # 其他元数据...
    }
    """
    # 1. 加载原始数据
    # 2. 转换格式（必要时做「裁剪」「重命名规范化」等预处理）
    # 3. 验证路径（可选，但强烈建议）
    # 4. 写入 JSONL 文件（单行 + pretty 两个版本）
    # 5. 输出统计信息（总样本数 / 缺失视频 / 各任务分布等）
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XXX to standard JSONL format')
    parser.add_argument('--input', type=str, default='/path/to/input')
    parser.add_argument('--output', type=str, default='/path/to/output.jsonl')
    parser.add_argument('--data-root', type=str, default='/path/to/data/')
    parser.add_argument('--no-verify', action='store_true')
    
    args = parser.parse_args()
    convert_xxx_to_jsonl(args.input, args.output, args.data_root, not args.no_verify)
```

运行转换：

```bash
cd playground
mkdir -p /path/to/dataset/annotation
python3 convert_xxx.py --no-verify
```

### 步骤 2.1: 长视频 / 带时间戳任务的裁剪策略（OVOBench / StreamingBench 等）

对于「长视频 + 每道题有明确时间戳」的 benchmark（例如 **OVOBench**、**StreamingBench**），
**不要直接把整段长视频喂给模型**，而应在转换脚本中显式处理时间窗口：

- **典型模式 A：离线 chunk（推荐）**
  - 每个问题一条样本，每条样本对应一个「已裁剪好」的视频片段：
    - OVOBench：`chunked_videos/{id}.mp4` / `{id}_{j}.mp4`，基于 `realtime` 或 `test_info.realtime`；
    - StreamingBench-Real/Omni/SQA：`chunked_*_60/<原视频名>_start_end.mp4`，基于 `time_stamp` 和 `context_time`（例如 60 秒）。
  - 在 JSONL 中：
    - `VideoPath` 直接写 **片段路径**（相对 `data_root`），例如：
      - `chunked_videos/123.mp4`
      - `chunked_omni_60/sample_2_Misleading_Context_Understanding_240_300.mp4`
  - 优点：
    - 与官方实现严格对齐（上下文窗口一致）；
    - 评测时不再做在线 ffmpeg 裁剪，复现更稳定。

- **典型模式 B：按时间字段在线裁剪（高级用法）**
  - 不在转换阶段生成片段，只在 JSONL 中记录：
    - `VideoPath`: 原始长视频路径；
    - `time_stamp` / `time_range` / `start_sec` / `end_sec` 等字段；
  - 在视频加载逻辑（如 `load_video` / `encode_video`）里使用这些时间段只采样局部帧。
  - 目前本仓库主线多使用 **模式 A（离线 chunk）**，模式 B 仅作为扩展思路。

**实践经验（坑）：**

- **文件名一致性非常关键**：
  - 官方标注里可能有 `sample_18_Scene_Understanding_1-25.mp4`，而真实磁盘上只有 `sample_18_Scene_Understanding.mp4`。
  - 建议在 `convert_xxx.py` 中做一层「规范化」逻辑（如 `normalize_video_name`）来映射到真实文件：
    - 先直接检查原名是否存在；
    - 不存在时，尝试去掉 `_1-25` / `_26-50` 等后缀再检查；
    - 将映射过程打印出来，方便排查。

- **长视频时间窗口选择**：
  - OVOBench：通常从 0 到 `realtime`；
  - StreamingBench：严格遵循官方，使用 `[timestamp - context_time, timestamp]`（默认 60 秒）。

### 步骤 3: 从视频提取音频 ⚠️ 重要

**`load_av` 操作需要音频文件存在！** 框架会自动从视频同目录查找同名音频：
- `xxx.mp4` → 查找 `xxx.wav` / `xxx.mp3` / `xxx.m4a` / `xxx.flac`

如果视频目录没有音频文件，需要提前提取：

```bash
# 方法 1: 在转换时提取（推荐）
#   - 如果采用「离线 chunk」策略，建议对「片段」提取音频，而不是对原始长视频提取
#   - 即：先裁剪再提取：full.mp4 → chunk_x_y.mp4 → chunk_x_y.wav
python3 convert_xxx.py --extract-audio --workers 8

# 方法 2: 批量提取工具
python3 playground/extract_audio_batch.py \
    --data-dir /path/to/dataset \
    --dry-run  # 先扫描统计

python3 playground/extract_audio_batch.py \
    --data-dir /path/to/dataset \
    --workers 8  # 执行提取

# 方法 3: 从 JSONL 读取视频路径提取
python3 playground/extract_audio_batch.py \
    --jsonl /path/to/annotation.jsonl \
    --data-dir /path/to/dataset \
    --workers 8
```

提取参数：
- `--workers`: 并行数（默认 4，大数据集建议 8-16）
- `--sample-rate`: 采样率（默认 16000）
- `--dry-run`: 仅统计，不执行

### 步骤 4: 放置数据文件

将数据放到 `data/omni/raw_hf/` 对应子目录下（已通过 `scripts/hf_download.py` 下载的数据会自动放到正确位置）：

```bash
# 如果数据集已上传到 HF，用户可通过下载脚本自动获取
python scripts/hf_download.py --datasets xxx_dataset --output_dir ./data

# 开发阶段也可手动放置
cp -r /path/to/your/DatasetName ./data/omni/raw_hf/dataset-name
```

### 步骤 5: 添加数据集配置（DatasetConfig）

在 `o_e_Kit/utils/args/dataset_args.py` 中添加配置：

```python
# 在 DATASET_REGISTRY 列表中添加
DatasetConfig(
    name="xxx_dataset",                    # 内部名称（用于 --eval_xxx_dataset）
    display_name="XXX Dataset",            # 显示名称
    category="omni",                       # 类别: audio, video, omni
    subcategory="qa",                      # 子类别: qa, asr, caption, cls
    paths={
        "data_prefix_dir": "./data/omni/raw_hf/xxx-dataset/",
        "annotation_path": "./data/omni/raw_hf/xxx-dataset/annotation/xxx.jsonl"
    },
    description="XXX Dataset: N samples for task description"
),
```

> **多 split / 多子基准场景（例如 StreamingBench-Real/Omni/SQA）**
>
> 为什么要拆成多个 dataset？
> - **不同子基准需要不同的 prompt 模板 / generation 配置**：  
>   - 如 StreamingBench 中，Real 只是视频 MCQ，Omni 需要强调音频，SQA 还需要额外的 `{sqa_context}` 文本上下文。  
>   - 在本框架里，一个 `dataset_name` 对应 `omni_generation_configs.json` 里一条唯一的 `user_prompt`，因此不能把语义差异很大的子任务塞到同一个 dataset 下。
> - 评测和统计维度可能不同：  
>   - 不同子基准可能希望单独汇报 overall 分数或单独分组。  
>
> 实践上，可以为每个子基准注册一个独立的 `DatasetConfig`，共用 `data_prefix_dir`，但使用不同的 `annotation_path` 和生成配置。
> - 例如：
>
> ```python
> DatasetConfig(
>     name="streamingbench_real",
>     paths={
>         "data_prefix_dir": "./data/omni/raw_hf/streamingbench/",
>         "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_real.jsonl"
>     },
>     ...
> ),
> DatasetConfig(
>     name="streamingbench_omni",
>     paths={
>         "data_prefix_dir": "./data/omni/raw_hf/streamingbench/",
>         "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_omni.jsonl"
>     },
>     ...
> ),
> DatasetConfig(
>     name="streamingbench_sqa",
>     paths={
>         "data_prefix_dir": "./data/omni/raw_hf/streamingbench/",
>         "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_sqa.jsonl"
>     },
>     ...
> ),
> ```

### 步骤 6: 添加数据集加载逻辑

在 `o_e_Kit/utils/dataset_loader.py` 的 `load_dataset` 函数中添加：

```python
elif dataset_name == "xxx_dataset":
    dataset = OmniEvalDataset(
        annotation_path=args.xxx_dataset_annotation_path,
        data_prefix_dir=args.xxx_dataset_data_prefix_dir,
        dataset_name=dataset_name
    )
```

### 步骤 7: 添加评测逻辑（evaluate_omni_datasets / eval.py）

在 `o_e_Kit/utils/evaluation_runner.py` 的 `evaluate_omni_datasets` 函数中添加：

```python
# XXX Dataset 评估
if getattr(args, 'eval_xxx_dataset', False):
    dataset = load_dataset(args, "xxx_dataset")
    result['xxx_dataset'] = infer_and_evaluate(
        model, dataset, args.model_name, "xxx_dataset", time,
        answer_path=args.answer_path, batch_size=args.batchsize,
        generate_method=args.generate_method
    )
```

> **按任务类别分组统计（eval.py）**
>
> - 如果新数据集需要按某些元数据字段（如 `task` / `task_group` / `task_type` / `required_ability` 等）分组统计分数，
>   需要在 `o_e_Kit/utils/eval.py` 的 `group_fields` 中补充：
>
> ```python
> group_fields = {
>     ...
>     'ovobench': ['task', 'task_group'],  # 小任务 & 大类
>     'streamingbench_real': ['task_type', 'required_ability', 'video_categories'],
>     'streamingbench_omni': ['task_type', 'required_ability', 'video_categories'],
>     'streamingbench_sqa': ['task_type', 'required_ability', 'video_categories'],
> }
> ```

### 步骤 8: 添加生成配置（omni_generation_configs.json）

在 `o_e_Kit/configs/omni_generation_configs.json` 中添加：

```json
{
    "mcq": {
        "xxx_dataset": {
            "user_prompt": "{media}\n{question}\n{options}\nPlease select the correct answer from the options above. Only respond with the letter.",
            "system_prompt": "",
            "max_tokens": 128,
            "max_frames": 64,      // 最大帧数（长视频可增大）
            "max_fps": 1.0,        // 采样帧率（长视频可降低）
            "load_av": true,       // 是否加载音视频
            "keep_placeholder": false,
            "interleave_fps": 1.0  // 音视频交错频率（段/秒），0 表示不交错
        }
    }
}
```

配置说明：
- `max_frames`: 最大采样帧数，长视频可设为 128
- `max_fps`: 采样帧率，长视频可设为 0.5（StreamingBench / LVBench 等长视频推荐降采样）
- `load_av`: 是否从视频提取音频。为 `true` 时，框架会从 `video_path` 同目录查找同名音频。
- `keep_placeholder`: 是否保留媒体占位符（如 `<audio_1>`）。对于带占位符的多音频任务（UNO-Bench、AV-Odyssey 等）使用。
- `interleave_fps`: 音频与帧的交错频率（段/秒）。例如 `1.0` 表示每秒切一段音频并交错到帧序列中；`0` 表示不交错，整段音频在所有帧之后。

## 验证集成

```bash
# 1. 验证配置加载
python3 -c "
from o_e_Kit.utils.args.dataset_args import DATASET_REGISTRY
for c in DATASET_REGISTRY:
    if 'xxx' in c.name:
        print(f'{c.name}: {c.paths}')
"

# 2. 快速验证（少量样本）
python3 eval_main.py --eval_xxx_dataset --max_sample_num 5

# 3. 完整评测
python3 eval_main.py --eval_xxx_dataset --max_sample_num 100
```

## 已集成数据集示例

| 数据集 | 样本数 | 类型 | 转换脚本 |
|--------|--------|------|----------|
| Daily-Omni | 1,197 | MCQ | convert_daily_omni.py |
| OmniBench | 1,142 | MCQ | convert_omnibench.py |
| WorldSense | 3,172 | MCQ | (已有jsonl) |
| AV-Odyssey | 4,555 | MCQ | convert_av_odyssey.py |
| UNO-Bench MC | 1,000 | MCQ | convert_unobench_mc.py |
| Video-Holmes | 1,837 | MCQ | (已有jsonl) |
| AVUT-Benchmark Human | 1,734 | MCQ | convert_avut_benchmark.py |
| AVUT-Benchmark Gemini | 9,874 | MCQ | convert_avut_benchmark.py |
| LVBench | 1,549 | MCQ | convert_lvbench.py |
| OVO-Bench (Omni) | - | MCQ/QA | convert_ovobench_omni.py |
| StreamingBench-Real (Offline) | - | MCQ | convert_streamingbench_real.py |
| StreamingBench-Omni (Offline) | - | MCQ | convert_streamingbench_omni.py |
| StreamingBench-SQA (Offline) | - | MCQ | convert_streamingbench_sqa.py |

## 统一 JSONL 格式规范

### MCQ (多选题)

```json
{
  "VideoPath": "path/to/video.mp4",
  "WavPath": "",
  "ImagePath": "",
  "dataset_type": "mcq",
  "dataset_name": "dataset_name",
  "question": "问题文本",
  "choices": ["选项A内容", "选项B内容", "选项C内容", "选项D内容"],
  "gt_answer": "A"
}
```

### Open QA (开放问答)

```json
{
  "VideoPath": "path/to/video.mp4",
  "WavPath": "",
  "dataset_type": "open_qa",
  "dataset_name": "dataset_name",
  "question": "问题文本",
  "gt_answer": "完整答案文本"
}
```

### 路径字典格式（多媒体）

用于支持多个音频/图片/视频的场景（如 UNO-Bench, AV-Odyssey）：

```json
{
  "audio_paths_dict": {"<audio_1>": "path/to/audio1.wav", "<audio_2>": "path/to/audio2.wav"},
  "image_paths_dict": {"<image_1>": "path/to/image1.jpg"},
  "video_paths_dict": {"<video_1>": "path/to/video1.mp4"},
  "question": "Listen to <audio_1> and look at <image_1>, ...",
  "choices": [...],
  "gt_answer": "B"
}
```

### 长视频 / 时间戳任务推荐字段（可选但强烈建议）

- **OVOBench**：
  - `task`: 细粒度任务类型（EPM / ASI / HLD / STU / OJR / ATR / ACR / OCR / FPD / REC / SSR / CRR）
  - `task_group`: 上层任务大类（如 `Backward Tracing`、`Real-Time Visual Perception` 等）
  - `realtime`: 官方提供的时间字段，用于裁剪视频或统计。

- **StreamingBench-Real/Omni**：
  - `task_type`: 官方任务类型（Clips Summarize / Object Recognition / ...）
  - `required_ability`: 能力标签（episodic memory / working memory / ...）
  - `video_categories`: 视频场景类别（preparation_of_meals / playing_card / ...）
  - `time_range`: 原始标注给出的时间段字符串（"[0:00:00 - 0:01:00]"）
  - `time_stamp`: 当前问题时间戳（"HH:MM:SS"）

- **StreamingBench-SQA（顺序问答）**：
  - 除上述字段外，建议额外添加：
    - `sqa_context`: 文本化的历史问答上下文，用于在 prompt 中复刻官方 SQA 逻辑，例如：
      - `"Here are the contextual information ... At timestamp 00:00:36, the following question and answer occurred: Question: ...; Options: ...; Answer: A; ..."`

## 注意事项

1. **选项格式**: `choices` 列表中不带字母前缀，字母由模型推理时动态添加
2. **答案格式**: MCQ 的 `gt_answer` 只存字母（A/B/C/D）
3. **路径验证**: 转换时可选验证媒体文件是否存在
4. **输出双版本**: 同时输出 `.jsonl`（单行）和 `.pretty.jsonl`（格式化）
5. **数据目录**: 评测数据通过 `scripts/hf_download.py` 从 HuggingFace 下载，开发阶段也可手动放置
6. **音频提取** ⚠️: 
   - `load_av=true` 时，框架会自动查找视频同目录的 `.wav` 文件
   - 音频查找规则: `xxx.mp4` → `xxx.wav` / `xxx.mp3` / `xxx.m4a` / `xxx.flac`
   - 如果音频不存在，**必须提前提取**，否则 `load_av` 无音频可加载
   - 使用 `playground/extract_audio_batch.py` 批量提取
   - 或在转换脚本中加 `--extract-audio` 参数
7. **文件名一致性** ⚠️:
   - 确保标注中的 `VideoPath` 与真实磁盘上的文件名一致（包括大小写和后缀）。
   - 对于官方标注中带有分段后缀（如 `_1-25` / `_26-50`）而真实文件只有一个整体视频的情况，要么：
     - 在下载/解压阶段按官方说明生成对应分段文件；要么
     - 在 `convert_xxx.py` 里添加「规范化」逻辑（参见 StreamingBench 集成中的 `normalize_video_name` 实践）。
   - 转换完成后，仔细检查统计信息中的 `missing_video` / `missing_clip`，一旦不为 0 要优先排查。

