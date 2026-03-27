# Supported Evaluation Metrics

This document provides a detailed description of the evaluation metrics currently supported in Omni-Eval Kit (o_e_Kit), including their definitions, computation methods, and how to interpret them in evaluation reports.

## Table of Contents

1. [Speech Recognition Metrics (WER / CER / SER)](#1-speech-recognition-metrics-wer--cer--ser)
2. [Multiple-Choice Accuracy (MQA)](#2-multiple-choice-accuracy-mqa)
3. [Reference Answer QA Accuracy (RefQA)](#3-reference-answer-qa-accuracy-refqa)
4. [Open-Ended QA Scoring (OpenQA)](#4-open-ended-qa-scoring-openqa)
5. [LLM Judge Win Rate](#5-llm-judge-win-rate)
6. [Caption Metrics (BLEU / METEOR / CIDEr)](#6-caption-metrics-bleu--meteor--cider)
7. [Other Metrics](#7-other-metrics)

---

## 1. Speech Recognition Metrics (WER / CER / SER)

Provided by the `WER_Eval` class in `o_e_Kit/utils/wer_eval.py`. **Does not require LLM API.**

### WER (Word Error Rate)

-   **Definition**: Measures the number of **word-level** edit operations (substitutions, deletions, insertions) required to transform the model's prediction into the ground truth, divided by the total number of words in the ground truth.
-   **Formula**:
    \[ \text{WER} = \frac{S + D + I}{N} \times 100\% \]
    Where:
    -   `S`: Number of substituted words.
    -   `D`: Number of deleted words.
    -   `I`: Number of inserted words.
    -   `N`: Total number of words in the ground truth.
-   **Interpretation**: Lower WER indicates higher recognition accuracy.
-   **Applicable datasets**: gigaspeech_test, librispeech_*, commonvoice_en/fr, voxpopuli_en, fleurs_en, peoples_speech_test, tedlium3_test

### CER (Character Error Rate)

-   **Definition**: Similar to WER, but the computation unit is **characters** instead of words. Suitable for non-segmented languages (e.g., Chinese).
-   **Formula**:
    \[ \text{CER} = \frac{S' + D' + I'}{N'} \times 100\% \]
-   **Interpretation**: Lower CER indicates higher recognition accuracy.
-   **Applicable datasets**: wenetspeech_*, commonvoice_zh/yue, aishell1/2_test, kespeech_test, fleurs_zh

### SER (Sentence Error Rate)

-   **Definition**: The proportion of incorrectly recognized sentences. A sentence is considered incorrect if it contains at least one word or character error.
-   **Formula**:
    \[ \text{SER} = \frac{\text{Number of sentences with at least one error}}{\text{Total number of sentences}} \times 100\% \]

### Implementation Details

1.  **Text Normalization**: Before computation, the `text_normalization` module is called to perform lowercasing, punctuation removal, number format unification, and other preprocessing.
2.  **Dynamic Switching**: The `metric` parameter (`'wer'` or `'cer'`) specifies word-level or character-level tokenization.
3.  **Error Statistics**: S, D, I are computed using the Levenshtein distance algorithm.

---

## 2. Multiple-Choice Accuracy (MQA)

Provided by the `MQAEvaluator` class in `o_e_Kit/utils/metrics/evaluator_mqa.py`. **Requires LLM API (as fallback).**

-   **Definition**: Extracts option letters (A-J) from the model's free-text answer, compares with the ground truth, and computes accuracy.
-   **Evaluation pipeline**:
    1. **Rule matching**: Extracts answer letters via regex and template matching
    2. **Sentence Transformer**: Computes cosine similarity between prediction text and each option (optional)
    3. **LLM extraction**: Calls `gpt-4o-mini` to extract an option letter from the model output
-   **Computation**: Accuracy = correct samples / total samples
-   **Interpretation**: Higher is better. Reports also show `rule_eval_count` (rule-matched) and `llm_eval_count` (LLM-evaluated) to reveal the evaluation method distribution.
-   **Applicable datasets**: voicebench_mmsu, voice_cmmlu, omnibench, daily_omni, videomme, av_odyssey, and 20+ other multiple-choice datasets

---

## 3. Reference Answer QA Accuracy (RefQA)

Provided by the `RefQAEvaluator` class in `o_e_Kit/utils/metrics/evaluator_refqa.py`. **Requires LLM API (as fallback).**

-   **Definition**: Determines whether the model's free-text answer is semantically consistent with a reference answer.
-   **Evaluation pipeline**:
    1. **Rule matching**: Substring containment, exact match, Yes/No determination
    2. **LLM judgment**: Calls `gpt-4o-mini` to determine if the prediction matches the reference answer (yes/no)
-   **Computation**: Accuracy = correct samples / total samples
-   **Applicable datasets**: voicebench_sdqa, voicebench_bbh, audio_web_questions, audio_trivia_qa, vocalsound, meld

---

## 4. Open-Ended QA Scoring (OpenQA)

Provided by the `OpenQAEvaluator` class in `o_e_Kit/utils/metrics/evaluator_openqa.py`. **Requires LLM API.**

-   **Definition**: Quality scoring for open-ended answers that have no standard reference answer.
-   **Evaluation pipeline**: All samples are evaluated via LLM (`gpt-4o-mini`), with no rule-matching stage.
-   **Computation**: LLM assigns each answer a score from 1 to 5; the final report shows the average.
-   **Interpretation**: Higher scores indicate better answer quality (5 is the maximum).
-   **Applicable datasets**: voicebench_alpacaeval, voicebench_commoneval, voicebench_wildvoice

---

## 5. LLM Judge Win Rate

Provided by `o_e_Kit/utils/metrics/evaluator_livesports3k_llm_judge.py`. **Requires LLM API (`gpt-4o`).**

-   **Definition**: Compares model output against a baseline model (GPT-4o) via A/B comparison, with an LLM judging which is better.
-   **Evaluation pipeline**:
    1. Each sample is judged in two rounds (AB order and BA order) to eliminate position bias
    2. Per-round scoring: model wins = 1 point, tie = 0.5 points, baseline wins = 0 points
    3. The two rounds are averaged
-   **Computation**: Win rate = average score across all samples
-   **Interpretation**: 50% means on par with the baseline; above 50% means the model outperforms the baseline.
-   **Applicable datasets**: livesports3k_cc
-   **Reference**: [LiveCC official implementation](https://github.com/showlab/livecc/blob/main/evaluation/livesports3kcc/llm_judge.py)

---

## 6. Caption Metrics (BLEU / METEOR / CIDEr)

Provided by the `CaptionEvaluator` class in `o_e_Kit/utils/metrics/evaluator_caption.py`. **Does not require LLM API.**

-   **BLEU**: Measures n-gram precision between generated and reference text
-   **METEOR**: Considers precision, recall, and word order for matching
-   **CIDEr**: TF-IDF weighted n-gram similarity, designed specifically for captioning tasks
-   **SPIDEr**: Average of CIDEr and SPICE
-   **Applicable datasets**: audiocaps_test, clothocaption_test, wavcaps_*, covost2_zh_en/en_zh

---

## 7. Other Metrics

### Safety Evaluation

Provided by `SafetyEvaluator`. **Does not require LLM API.** Checks whether the model correctly refuses harmful requests (0-1 scoring).

### Instruction Following Evaluation

Provided by `InstructionFollowingEvaluator`. **Does not require LLM API.** Evaluates whether the model follows specific format/content constraints.

### StreamingBench

Provided by `StreamingBenchEval`. **Does not require LLM API.** Evaluates accuracy on streaming video understanding tasks.

### OV-AVEL Event Localization

Provided by `OVAVELEvaluator`. **Does not require LLM API.** Evaluates frame-level accuracy, segment-level F1, and event-level F1.

---

## How to Extend

To add new evaluation metrics, see [How to Contribute a New Evaluation Method](../development/CONTRIBUTING_EVALS.md).

## Related Documentation

-   **[LLM Evaluation Configuration Guide](../guides/LLM_EVALUATION.md)** — How to configure LLM API for metrics that require it
