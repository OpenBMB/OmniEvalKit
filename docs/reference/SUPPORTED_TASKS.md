# Supported Evaluation Task Types

This document clearly lists the evaluation tasks supported by the current version of Omni-Eval Kit (o_e_Kit).

Our framework can flexibly combine different input and output modalities. The following evaluation task pipelines have been implemented and verified:

### 1. Speech + Text Input / Text Output

This is the typical speech question answering (Audio QA) or speech instruction understanding task.

-   **Input**:
    -   An audio clip (`.wav`)
    -   An associated text question or instruction (`question` field in `.jsonl`)
-   **Output**:
    -   Model-generated text answer (`answer` field in `.jsonl`)
-   **Typical evaluation metrics**:
    -   Word Error Rate (WER)
    -   Character Error Rate (CER)
    -   Keyword Matching

### 2. Speech + Text Input / Speech Output

This is a multi-modal dialogue task with speech synthesis capability.

-   **Input**:
    -   An audio clip (`.wav`)
    -   An associated text question or instruction (`.jsonl`)
-   **Output**:
    -   Model-generated speech response (saved as `.wav` file)
-   **Typical evaluation metrics**:
    -   **Objective metrics**: e.g., Mel-Cepstral Distortion (MCD) computed by acoustic models.
    -   **Subjective metrics (model scoring)**: e.g., using MOSNet (Mean Opinion Score) to predict the naturalness and quality of generated speech.

### 3. Video + Text Input / Text Output

This is the typical video content understanding or video question answering (Video QA) task.

-   **Input**:
    -   A video clip (`.mp4`)
    -   An associated text question (`.jsonl`)
-   **Output**:
    -   Model-generated text answer (`.jsonl`)
-   **Typical evaluation metrics**:
    -   Accuracy
    -   VQA Score

### 4. Video + Text + Speech Input / Text Output

This is the most comprehensive omni-modal understanding task, requiring the model to process all three mainstream modalities simultaneously.

-   **Input**:
    -   A video clip (`.mp4`)
    -   An associated text question (`.jsonl`)
    -   An associated user voice instruction or follow-up question (`.wav`)
-   **Output**:
    -   Model-generated unified text answer (`.jsonl`)
-   **Typical evaluation metrics**:
    -   Accuracy
    -   Task-specific metrics related to multi-modal fusion understanding.

This list will be updated as the project evolves.
