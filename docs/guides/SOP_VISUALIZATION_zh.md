# 评测结果可视化 SOP (Standard Operating Procedure)

本文档介绍如何使用 Omni-Eval Kit 的可视化工具生成评测报告和训练曲线。

## 📋 目录

- [概述](#概述)
- [工具介绍](#工具介绍)
- [快速开始](#快速开始)
- [HTML 报告（推荐）](#html-报告推荐)
- [详细用法](#详细用法)
- [输出示例](#输出示例)
- [常见问题](#常见问题)

---

## 概述

可视化工具支持以下功能：

| 功能 | 描述 |
|------|------|
| **ASR 曲线** | 绘制 WER/CER 训练曲线（越低越好） |
| **Omni 曲线** | 绘制多模态准确率曲线（越高越好） |
| **Omni Duplex 曲线** | 绘制全双工对话评测曲线 |
| **版本对比** | 对比 Decay V2 vs V3 等不同训练版本 |
| **基线对比** | 与 Qwen3-Omni-30B 等基线模型对比 |

---

## 工具介绍

可视化工具位于 `o_e_Kit/utils/visualizer/` 目录下：

```
o_e_Kit/utils/visualizer/
├── training_curve.py     # 核心绘图工具
├── generate_reports.py   # 批量报告生成
├── results_viewer.py     # Web 结果查看器
└── README.md
```

### 依赖安装

```bash
pip install matplotlib numpy
# 可选：用于 Web 查看器
pip install flask openpyxl
```

---

## 快速开始

### 方式一：使用预定义配置（推荐）

```bash
cd omnievalkit

# 生成所有报告（ASR + Omni + Duplex + 对比图）
python -m o_e_Kit.utils.visualizer.generate_reports \
    --asr-dir /path/to/asr/results \
    --omni-dir /path/to/omni/results \
    --output-dir ./reports
```

### 方式二：单独生成某类报告

```bash
# 只生成 ASR 报告
python -m o_e_Kit.utils.visualizer.training_curve \
    --type asr \
    --results-dir /path/to/asr/results \
    --output asr_training_curve.pdf

# 只生成 Omni 报告
python -m o_e_Kit.utils.visualizer.training_curve \
    --type omni \
    --results-dir /path/to/omni/results \
    --output omni_training_curve.pdf \
    --baseline qwen3_omni_30b
```

### 方式三：生成对比图

```bash
# V2 vs V3 对比
python -m o_e_Kit.utils.visualizer.training_curve \
    --type omni \
    --results-dir /path/to/v2_results /path/to/v3_results \
    --labels "Decay V2" "Decay V3" \
    --output v2_vs_v3_comparison.pdf
```

---

## HTML 报告（推荐）

生成可直接在浏览器查看的静态 HTML 报告，包含交互式图表和数据表格。

### 一键生成

```bash
# 使用快捷脚本
./scripts/update_viz.sh --output ./viz_output

# 或直接运行
cd o_e_Kit/utils/visualizer
python generate_html_report.py \
    --asr-dir /path/to/asr/results \
    --omni-dir /path/to/omni/results \
    --output ./viz_output
```

### 查看报告

报告目录: `<output>/eval_report/`

```
eval_report/
├── index.html          # 主入口（完整报告）
├── asr.html            # ASR 单独页面
├── omni.html           # Omni 单独页面  
├── duplex.html         # Duplex 单独页面
├── comparison.html     # V2 vs V3 对比
└── report_YYYYMMDD_HHMMSS.html  # 带时间戳的历史版本
```

直接在浏览器中打开 `index.html` 即可查看完整报告。

### 定期更新

每次有新的评测结果时，运行：

```bash
./scripts/update_viz.sh
```

报告会自动刷新，包含最新的：
- ASR 训练曲线和数据表格
- Omni 训练曲线和数据表格
- Omni Duplex 训练曲线
- Decay V2 vs V3 对比图

### HTML 报告特性

| 特性 | 说明 |
|------|------|
| 📊 嵌入式图表 | 图表以 base64 嵌入，无需外部依赖 |
| 📋 数据表格 | 每个 step 的详细分数，带颜色标记 |
| 📱 响应式设计 | 支持手机/平板/桌面设备 |
| 🎨 深色主题 | 护眼的深色界面 |
| 🔄 一键更新 | 运行脚本即可刷新 |

---

## 详细用法

### training_curve.py 参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--type, -t` | ✅ | 评测类型：`asr`, `omni`, `omni_duplex` |
| `--results-dir, -d` | ✅ | 结果目录（可多个用于对比） |
| `--labels, -l` | | 各目录的标签 |
| `--output, -o` | | 输出文件路径 (默认: `training_curve.png`) |
| `--title` | | 图表标题 |
| `--baseline, -b` | | 基线模型：`qwen3_omni_30b` |

### generate_reports.py 参数

| 参数 | 说明 |
|------|------|
| `--config, -c` | 使用预定义配置：`final_monitor_v1`, `final_stable_v1` |
| `--asr-dir` | ASR 结果目录 |
| `--omni-dir` | Omni 结果目录 |
| `--omni-duplex-dir` | Omni Duplex 结果目录 |
| `--v2-dir` | V2 结果目录（用于对比） |
| `--v3-dir` | V3 结果目录（用于对比） |
| `--output-dir, -o` | 输出目录 (默认: `./reports`) |
| `--skip-asr` | 跳过 ASR 报告 |
| `--skip-omni` | 跳过 Omni 报告 |
| `--skip-duplex` | 跳过 Omni Duplex 报告 |
| `--skip-comparison` | 跳过对比报告 |

---

## 输出示例

### ASR 曲线图

ASR 报告包含 4 个子图：
1. **Chinese ASR**: AISHELL-1/2, WenetSpeech, CommonVoice-ZH 等
2. **English ASR (LibriSpeech)**: test-clean/other, dev-clean/other
3. **English ASR (Other)**: GigaSpeech, TED-LIUM, VoxPopuli 等
4. **Average**: 中/英文及总平均 WER/CER

### Omni 曲线图

Omni 报告包含 4 个子图：
1. **Video Understanding**: VideoMME, OVOBench, Video-Holmes 等
2. **Audio-Video & Omni**: AV-Odyssey, AVUT, OmniBench, WorldSense
3. **Streaming & Other**: StreamingBench, Omni-OpenQA
4. **Total Average**: 总平均准确率（含 Qwen3-Omni 基线）

### 对比图

对比图使用不同的线型区分版本：
- 实线 (`-`): 第一个版本
- 虚线 (`--`): 第二个版本
- 点线 (`:`): 第三个版本

---

## 目录结构要求

评测结果目录需要满足以下结构：

```
results_dir/
├── experiment_name_step_1000/
│   └── asr/  (或 omni/ 或 omni_duplex/)
│       ├── result_dataset1.json
│       ├── result_dataset2.json
│       └── ...
├── experiment_name_step_2000/
│   └── ...
└── ...
```

其中 `result_*.json` 文件格式：

```json
{
    "dataset_name": "gigaspeech_test",
    "score": 8.9,
    "model_name": "model_name",
    "evaluation_time": "...",
    "job_id": "12345"
}
```

---

## Web 结果查看器

除了静态报告，还可以使用 Web 查看器交互式浏览结果：

```bash
python -m o_e_Kit.utils.visualizer.results_viewer \
    --results-dir /path/to/your/results \
    --port 6000
```

功能特性：
- 📁 智能分级浏览结果目录
- 📊 自动识别并汇总评测结果
- ❌ 错题筛选（只看 match=false）
- 🏷️ 动态字段筛选
- 📥 导出 Excel 汇总表
- ⌨️ 键盘快捷键支持

---

## 常见问题

### Q: 图表中文显示为方块？

安装中文字体或使用 DejaVu Sans：
```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
```

### Q: 如何添加新的基线模型？

编辑 `training_curve.py` 中的 `BASELINES` 字典：
```python
BASELINES = {
    'my_baseline': {
        'dataset1': 70.5,
        'dataset2': 65.3,
        ...
    }
}
```

### Q: 如何自定义数据集分类？

编辑 `training_curve.py` 中的 `ASR_DATASETS` 或 `OMNI_DATASETS`：
```python
OMNI_DATASETS = {
    'video': ['videomme', 'ovobench', ...],
    'audio_video': ['av_odyssey', ...],
    ...
}
```

---

## 预定义配置

### 自定义配置示例

```python
{
    'asr_dir': '/path/to/asr/results',
    'omni_dir': '/path/to/omni/results',
    'omni_duplex_dir': '/path/to/duplex/results',
    'output_dir': './reports',
}
```

使用方法：
```bash
python -m o_e_Kit.utils.visualizer.generate_reports \
    --asr-dir /path/to/asr/results \
    --omni-dir /path/to/omni/results \
    --output-dir ./reports
```

---

## 联系方式

如有问题，请联系项目维护者或在 GitHub 提交 Issue。

