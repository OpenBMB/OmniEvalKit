# Evaluation Result Visualization SOP (Standard Operating Procedure)

This document describes how to use Omni-Eval Kit's visualization tools to generate evaluation reports and training curves.

## Table of Contents

- [Overview](#overview)
- [Tools](#tools)
- [Quick Start](#quick-start)
- [HTML Report (Recommended)](#html-report-recommended)
- [Detailed Usage](#detailed-usage)
- [Output Examples](#output-examples)
- [FAQ](#faq)

---

## Overview

The visualization tools support the following features:

| Feature | Description |
|---------|-------------|
| **ASR Curves** | Plot WER/CER training curves (lower is better) |
| **Omni Curves** | Plot multi-modal accuracy curves (higher is better) |
| **Omni Duplex Curves** | Plot full-duplex dialogue evaluation curves |
| **Version Comparison** | Compare different training versions (e.g., Decay V2 vs V3) |
| **Baseline Comparison** | Compare against baseline models like Qwen3-Omni-30B |

---

## Tools

Visualization tools are located in the `o_e_Kit/utils/visualizer/` directory:

```
o_e_Kit/utils/visualizer/
├── training_curve.py     # Core plotting tool
├── generate_reports.py   # Batch report generation
├── results_viewer.py     # Web results viewer
└── README.md
```

### Dependencies

```bash
pip install matplotlib numpy
# Optional: for web viewer
pip install flask openpyxl
```

---

## Quick Start

### Option 1: Use Predefined Configuration (Recommended)

```bash
cd omnievalkit

# Generate all reports (ASR + Omni + Duplex + comparison)
python -m o_e_Kit.utils.visualizer.generate_reports \
    --asr-dir /path/to/asr/results \
    --omni-dir /path/to/omni/results \
    --output-dir ./reports
```

### Option 2: Generate a Specific Report Type

```bash
# Generate ASR report only
python -m o_e_Kit.utils.visualizer.training_curve \
    --type asr \
    --results-dir /path/to/asr/results \
    --output asr_training_curve.pdf

# Generate Omni report only
python -m o_e_Kit.utils.visualizer.training_curve \
    --type omni \
    --results-dir /path/to/omni/results \
    --output omni_training_curve.pdf \
    --baseline qwen3_omni_30b
```

### Option 3: Generate Comparison Charts

```bash
# V2 vs V3 comparison
python -m o_e_Kit.utils.visualizer.training_curve \
    --type omni \
    --results-dir /path/to/v2_results /path/to/v3_results \
    --labels "Decay V2" "Decay V3" \
    --output v2_vs_v3_comparison.pdf
```

---

## HTML Report (Recommended)

Generate static HTML reports viewable directly in a browser, with interactive charts and data tables.

### One-Click Generation

```bash
# Using the shortcut script
./scripts/update_viz.sh --output ./viz_output

# Or run directly
cd o_e_Kit/utils/visualizer
python generate_html_report.py \
    --asr-dir /path/to/asr/results \
    --omni-dir /path/to/omni/results \
    --output ./viz_output
```

### Viewing Reports

Report directory: `<output>/eval_report/`

```
eval_report/
├── index.html          # Main entry (full report)
├── asr.html            # ASR standalone page
├── omni.html           # Omni standalone page
├── duplex.html         # Duplex standalone page
├── comparison.html     # V2 vs V3 comparison
└── report_YYYYMMDD_HHMMSS.html  # Timestamped historical version
```

Open `index.html` in your browser to view the full report.

### Periodic Updates

Whenever new evaluation results are available, run:

```bash
./scripts/update_viz.sh
```

The report will automatically refresh, including the latest:
- ASR training curves and data tables
- Omni training curves and data tables
- Omni Duplex training curves
- Decay V2 vs V3 comparison charts

### HTML Report Features

| Feature | Description |
|---------|-------------|
| Embedded Charts | Charts embedded as base64, no external dependencies |
| Data Tables | Detailed scores per step, with color coding |
| Responsive Design | Supports mobile / tablet / desktop |
| Dark Theme | Eye-friendly dark interface |
| One-Click Update | Run the script to refresh |

---

## Detailed Usage

### training_curve.py Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--type, -t` | Yes | Evaluation type: `asr`, `omni`, `omni_duplex` |
| `--results-dir, -d` | Yes | Results directory (multiple directories for comparison) |
| `--labels, -l` | | Labels for each directory |
| `--output, -o` | | Output file path (default: `training_curve.png`) |
| `--title` | | Chart title |
| `--baseline, -b` | | Baseline model: `qwen3_omni_30b` |

### generate_reports.py Parameters

| Parameter | Description |
|-----------|-------------|
| `--config, -c` | Use predefined config: `final_monitor_v1`, `final_stable_v1` |
| `--asr-dir` | ASR results directory |
| `--omni-dir` | Omni results directory |
| `--omni-duplex-dir` | Omni Duplex results directory |
| `--v2-dir` | V2 results directory (for comparison) |
| `--v3-dir` | V3 results directory (for comparison) |
| `--output-dir, -o` | Output directory (default: `./reports`) |
| `--skip-asr` | Skip ASR report |
| `--skip-omni` | Skip Omni report |
| `--skip-duplex` | Skip Omni Duplex report |
| `--skip-comparison` | Skip comparison report |

---

## Output Examples

### ASR Curve Chart

ASR reports contain 4 subplots:
1. **Chinese ASR**: AISHELL-1/2, WenetSpeech, CommonVoice-ZH, etc.
2. **English ASR (LibriSpeech)**: test-clean/other, dev-clean/other
3. **English ASR (Other)**: GigaSpeech, TED-LIUM, VoxPopuli, etc.
4. **Average**: Chinese / English and overall average WER/CER

### Omni Curve Chart

Omni reports contain 4 subplots:
1. **Video Understanding**: VideoMME, OVOBench, Video-Holmes, etc.
2. **Audio-Video & Omni**: AV-Odyssey, AVUT, OmniBench, WorldSense
3. **Streaming & Other**: StreamingBench, Omni-OpenQA
4. **Total Average**: Overall average accuracy (with Qwen3-Omni baseline)

### Comparison Charts

Comparison charts use different line styles to distinguish versions:
- Solid line (`-`): First version
- Dashed line (`--`): Second version
- Dotted line (`:`): Third version

---

## Directory Structure Requirements

Evaluation result directories must follow this structure:

```
results_dir/
├── experiment_name_step_1000/
│   └── asr/  (or omni/ or omni_duplex/)
│       ├── result_dataset1.json
│       ├── result_dataset2.json
│       └── ...
├── experiment_name_step_2000/
│   └── ...
└── ...
```

Where `result_*.json` file format is:

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

## Web Results Viewer

In addition to static reports, you can use the web viewer to interactively browse results:

```bash
python -m o_e_Kit.utils.visualizer.results_viewer \
    --results-dir /path/to/your/results \
    --port 6000
```

Features:
- Hierarchical browsing of results directories
- Automatic detection and aggregation of evaluation results
- Error filtering (show only match=false)
- Dynamic field filtering
- Export to Excel summary
- Keyboard shortcuts

---

## FAQ

### Q: Chinese characters display as squares in charts?

Install Chinese fonts or use DejaVu Sans:
```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
```

### Q: How to add a new baseline model?

Edit the `BASELINES` dictionary in `training_curve.py`:
```python
BASELINES = {
    'my_baseline': {
        'dataset1': 70.5,
        'dataset2': 65.3,
        ...
    }
}
```

### Q: How to customize dataset categories?

Edit `ASR_DATASETS` or `OMNI_DATASETS` in `training_curve.py`:
```python
OMNI_DATASETS = {
    'video': ['videomme', 'ovobench', ...],
    'audio_video': ['av_odyssey', ...],
    ...
}
```

---

## Predefined Configurations

### Custom Configuration Example

```python
{
    'asr_dir': '/path/to/asr/results',
    'omni_dir': '/path/to/omni/results',
    'omni_duplex_dir': '/path/to/duplex/results',
    'output_dir': './reports',
}
```

Usage:
```bash
python -m o_e_Kit.utils.visualizer.generate_reports \
    --asr-dir /path/to/asr/results \
    --omni-dir /path/to/omni/results \
    --output-dir ./reports
```

---

## Contact

If you have questions, please contact the project maintainers or open an Issue on GitHub.
