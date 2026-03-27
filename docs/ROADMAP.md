# Roadmap & Design Philosophy

This document outlines the long-term development direction and core design principles of Omni-Eval Kit (o_e_Kit), providing clear guidance for future iterations and community contributions.

## 1. Feature Roadmap

Our goal is to build o_e_Kit into a comprehensive, user-friendly omni-modal evaluation framework. We will focus on supporting the following task types:

### 1.1 Evaluation Tasks

#### A. Simplex Tasks
The model receives one or more modality inputs and produces a single modality output.
- **Audio Understanding**: e.g., audio classification, speaker verification.
- **Speech Generation**: e.g., text-to-speech synthesis (TTS).
- **Omni-modal Understanding**: receives mixed inputs (audio, video, text) and produces text-based understanding and answers, e.g., VQA (Visual Question Answering).

#### B. Duplex Tasks
The model can receive and produce bidirectional, streaming multi-modal information, enabling conversation-like interaction.
- **Voice Interaction**: e.g., real-time voice dialogue systems.
- **Audio-Visual Interaction**: e.g., digital humans that understand and respond to video content, generating corresponding audio-visual streams.

### 1.2 Comprehensive Feature Support
To enable the above tasks, the framework will provide full support in the following dimensions:
- **Dataset Support**: Standardized data loading interfaces for all evaluation task types.
- **Model Inference Support**: Easy integration and switching between different models.
- **Result Persistence**: Unified, traceable result storage mechanisms.
- **Result Computation**: Built-in or extensible metric computation for different tasks.
- **In-depth Result Analysis**: Beyond aggregate scores like WER/CER, future versions will support finer-grained error analysis — e.g., automatically identifying frequent substitution pairs, common typos, or performance degradation under specific acoustic conditions (high noise, far-field).
- **Model Scoring**: For tasks like speech generation, support automated scoring with models such as MOSNet.
- **Multi-dimensional Visualization**: Rich, interactive visualization tools, including:
    - **Confusion Matrix**: Visually shows the words or phonemes the model most easily confuses.
    - **Error Distribution Histogram**: Analyzes error rate distributions by sentence length, SNR, and other dimensions.
    - **Web-based Result Browser**: A simple web interface for filtering, sorting, and inspecting evaluation results, displaying predicted audio, ground truth, and model output side by side for manual auditing.

## 2. Core Design Philosophy

To ensure robustness, extensibility, and ease of use, we uphold the following three core design principles:

### A. Unified Yet Flexible Model Inference
All model inference settings — **including prompts, temperature, max length, generation strategy (sampling/greedy), etc. — are encapsulated through mature classes**. Benefits:
- **Easy Management**: All model configurations are managed uniformly, making centralized inspection and modification convenient.
- **Guaranteed Alignment**: Ensures strict alignment of evaluation settings across different models and experiments.
- **High Customizability**: While providing standard encapsulation, flexible custom interfaces are preserved for developers.

### B. Pure and Transparent Data Formats
We believe that evaluation data and results should be simple, intuitive, and easy to inspect. Therefore, this framework **completely abandons any complex, encapsulated data types for evaluation data**.
- **Input**: Only the most basic, easily visualizable formats — `.wav` for audio, `.mp4` for video, `.jsonl` for text and structured annotations.
- **Output**: Model outputs are saved following the same principles.
This design philosophy eliminates black-box issues caused by complex data formats, ensuring transparency at every stage.

### C. Unified Evaluation Management
The ultimate purpose of this framework is to **unify the management of all evaluation strategies, integrating evaluation work across different teams, models, and tasks into a single unified framework**. This greatly improves team evaluation efficiency, ensures the credibility and comparability of results, and promotes knowledge accumulation and reuse.
