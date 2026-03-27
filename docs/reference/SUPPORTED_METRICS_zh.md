# 已支持的评测指标

本文档详细介绍了 Omni-Eval Kit (o_e_Kit) 中当前支持的评测指标 (Metrics)，包括它们的定义、计算方式以及如何在评测报告中解读这些指标。

## 目录

1. [语音识别指标（WER / CER / SER）](#1-语音识别指标wer--cer--ser)
2. [选择题准确率（MQA）](#2-选择题准确率mqa)
3. [参考答案问答准确率（RefQA）](#3-参考答案问答准确率refqa)
4. [开放问答评分（OpenQA）](#4-开放问答评分openqa)
5. [LLM Judge 胜率](#5-llm-judge-胜率)
6. [Caption 指标（BLEU / METEOR / CIDEr）](#6-caption-指标bleu--meteor--cider)
7. [其他指标](#7-其他指标)

---

## 1. 语音识别指标（WER / CER / SER）

由 `o_e_Kit/utils/wer_eval.py` 中的 `WER_Eval` 类提供。**不需要 LLM API。**

### WER (Word Error Rate - 词错误率)

-   **定义**: 计算模型预测结果相较于标准答案所需要进行的**词级别**的编辑操作（替换、删除、插入）的总数，再除以标准答案的总词数。
-   **计算公式**:
    \[ \text{WER} = \frac{S + D + I}{N} \times 100\% \]
    其中：
    -   `S`: 替换 (Substitutions) 的词数。
    -   `D`: 删除 (Deletions) 的词数。
    -   `I`: 插入 (Insertions) 的词数。
    -   `N`: 标准答案中的总词数。
-   **解读**: WER 的值越低，表示模型的识别准确率越高。
-   **适用数据集**: gigaspeech_test, librispeech_*, commonvoice_en/fr, voxpopuli_en, fleurs_en, peoples_speech_test, tedlium3_test

### CER (Character Error Rate - 字错误率)

-   **定义**: 与 WER 类似，但计算单元是**字符**而不是词。适用于非分词语言（如中文）。
-   **计算公式**:
    \[ \text{CER} = \frac{S' + D' + I'}{N'} \times 100\% \]
-   **解读**: CER 的值越低，表示模型的识别准确率越高。
-   **适用数据集**: wenetspeech_*, commonvoice_zh/yue, aishell1/2_test, kespeech_test, fleurs_zh

### SER (Sentence Error Rate - 句错误率)

-   **定义**: 被错误识别的句子的比例。只要一个句子中包含至少一个词或字的错误，该句子就被认为是错误的。
-   **计算公式**:
    \[ \text{SER} = \frac{\text{Number of Sentences with at least one error}}{\text{Total Number of Sentences}} \times 100\% \]

### 实现细节

1.  **文本正则化**: 计算前会调用 `text_normalization` 模块进行转小写、去除标点、统一数字格式等预处理。
2.  **动态切换**: 通过 `metric` 参数 (`'wer'` 或 `'cer'`) 指定按词还是按字切分。
3.  **错误统计**: 基于 Levenshtein 距离算法计算 S, D, I。

---

## 2. 选择题准确率（MQA）

由 `o_e_Kit/utils/metrics/evaluator_mqa.py` 中的 `MQAEvaluator` 类提供。**需要 LLM API（作为 Fallback）。**

-   **定义**: 从模型的自由文本回答中提取选项字母（A-J），与标准答案比较，计算准确率。
-   **评估流程**:
    1. **规则匹配**: 通过正则表达式和模板匹配提取答案字母
    2. **Sentence Transformer**: 计算预测文本与各选项的余弦相似度（可选）
    3. **LLM 提取**: 调用 `gpt-4o-mini`，要求其从模型输出中提取一个选项字母
-   **计算**: 准确率 = 正确样本数 / 总样本数
-   **解读**: 值越高越好。报告中还会显示 `rule_eval_count`（规则成功数）和 `llm_eval_count`（LLM 评估数），帮助了解评估方法分布。
-   **适用数据集**: voicebench_mmsu, voice_cmmlu, omnibench, daily_omni, videomme, av_odyssey 等 20+ 个选择题数据集

---

## 3. 参考答案问答准确率（RefQA）

由 `o_e_Kit/utils/metrics/evaluator_refqa.py` 中的 `RefQAEvaluator` 类提供。**需要 LLM API（作为 Fallback）。**

-   **定义**: 判定模型的自由文本回答是否与参考答案语义一致。
-   **评估流程**:
    1. **规则匹配**: 子串包含、精确匹配、Yes/No 判定
    2. **LLM 判定**: 调用 `gpt-4o-mini`，判定预测是否与参考答案一致（是/否）
-   **计算**: 准确率 = 正确样本数 / 总样本数
-   **适用数据集**: voicebench_sdqa, voicebench_bbh, audio_web_questions, audio_trivia_qa, vocalsound, meld

---

## 4. 开放问答评分（OpenQA）

由 `o_e_Kit/utils/metrics/evaluator_openqa.py` 中的 `OpenQAEvaluator` 类提供。**需要 LLM API。**

-   **定义**: 对没有标准参考答案的开放式回答进行质量评分。
-   **评估流程**: 所有样本均通过 LLM（`gpt-4o-mini`）评估，无规则匹配阶段。
-   **计算**: LLM 对每个回答给出 1-5 分评分，最终报告平均分。
-   **解读**: 分数越高表示回答质量越好（5 分满分）。
-   **适用数据集**: voicebench_alpacaeval, voicebench_commoneval, voicebench_wildvoice

---

## 5. LLM Judge 胜率

由 `o_e_Kit/utils/metrics/evaluator_livesports3k_llm_judge.py` 提供。**需要 LLM API（`gpt-4o`）。**

-   **定义**: 将模型输出与基线模型（GPT-4o）的输出进行 A/B 对比，由 LLM 判定哪个更好。
-   **评估流程**:
    1. 每个样本进行两轮评判（AB 顺序和 BA 顺序），消除位置偏差
    2. 每轮判定结果：模型胜 = 1 分，平局 = 0.5 分，基线胜 = 0 分
    3. 取两轮平均分
-   **计算**: 胜率 = 所有样本的平均得分
-   **解读**: 50% 表示与基线持平，高于 50% 表示模型优于基线。
-   **适用数据集**: livesports3k_cc
-   **参考**: [LiveCC 官方实现](https://github.com/showlab/livecc/blob/main/evaluation/livesports3kcc/llm_judge.py)

---

## 6. Caption 指标（BLEU / METEOR / CIDEr）

由 `o_e_Kit/utils/metrics/evaluator_caption.py` 中的 `CaptionEvaluator` 类提供。**不需要 LLM API。**

-   **BLEU**: 衡量生成文本与参考文本的 n-gram 精确度
-   **METEOR**: 综合考虑精确度、召回率和词序的匹配指标
-   **CIDEr**: 基于 TF-IDF 加权的 n-gram 相似度，专为描述任务设计
-   **SPIDEr**: CIDEr 和 SPICE 的平均值
-   **适用数据集**: audiocaps_test, clothocaption_test, wavcaps_*, covost2_zh_en/en_zh

---

## 7. 其他指标

### 安全评估

由 `SafetyEvaluator` 提供。**不需要 LLM API。** 检测模型是否正确拒绝有害请求（0-1 分制）。

### 指令遵循评估

由 `InstructionFollowingEvaluator` 提供。**不需要 LLM API。** 评估模型是否遵循了特定的格式/内容约束。

### StreamingBench

由 `StreamingBenchEval` 提供。**不需要 LLM API。** 评估流式视频理解任务的准确率。

### OV-AVEL 事件定位

由 `OVAVELEvaluator` 提供。**不需要 LLM API。** 评估帧级准确率、片段级 F1 和事件级 F1。

---

## 如何扩展

要添加新的评测指标，请参考 [如何贡献一个新的评测方法](../development/CONTRIBUTING_EVALS_zh.md)。

## 相关文档

-   **[LLM 评估配置指南](../guides/LLM_EVALUATION_zh.md)** - 如何配置 LLM API 以支持需要 API 的指标