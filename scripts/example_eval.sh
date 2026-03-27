#!/bin/bash
# =============================================================================
# OmniEvalKit 评测启动示例
#
# 用法:
#   1. 修改下方 "用户配置" 区域的变量
#   2. 取消注释你想运行的评测场景（默认运行 GigaSpeech ASR）
#   3. 执行: bash scripts/example_eval.sh
# =============================================================================

# ===================== 用户配置（请根据实际环境修改） =====================

MODEL_PATH="/path/to/your/model"          # 模型目录（HuggingFace 格式）
PT_PATH=""                                # .pt 检查点路径（可选，留空则不使用）
MODEL_TYPE="minicpmo"                     # 模型类型: minicpmo / qwen3_omni / gemini_omni / whisper
MODEL_NAME="my_eval"                      # 评测任务名称，用于结果目录命名
ANSWER_PATH="./results"                   # 结果保存目录
GPUS_PER_NODE=1                           # 使用的 GPU 数量
BATCH_SIZE=4                              # 推理批大小
MAX_SAMPLES=""                            # 最大样本数（留空则评测全量数据）

# ===================== 评测数据集选择（取消注释你需要的场景） =====================

# --- 场景 1: ASR 语音识别 ---
EVAL_DATASETS="--eval_gigaspeech_test"
# EVAL_DATASETS="--eval_aishell1_test"
# EVAL_DATASETS="--eval_librispeech_test_clean"
# EVAL_DATASETS="--eval_wenetspeech_test_net"
# EVAL_DATASETS="--eval_commonvoice_zh"
# EVAL_DATASETS="--eval_fleurs_en"

# --- 场景 2: Audio QA 语音问答 ---
# EVAL_DATASETS="--eval_voicebench_alpacaeval"
# EVAL_DATASETS="--eval_voicebench_commoneval"
# EVAL_DATASETS="--eval_voicebench_ifeval"
# EVAL_DATASETS="--eval_voicebench_bbh"

# --- 场景 3: Omni 多模态评测（视频/音视频理解，建议 batch_size=1） ---
# BATCH_SIZE=1
# EVAL_DATASETS="--eval_daily_omni --generate_method chat"
# EVAL_DATASETS="--eval_videomme --generate_method chat"
# EVAL_DATASETS="--eval_omnibench --generate_method chat"
# EVAL_DATASETS="--eval_worldsense --generate_method chat"
# EVAL_DATASETS="--eval_streamingbench_real --generate_method chat"

# --- 场景 4: Duplex 视频描述评测 ---
# BATCH_SIZE=1
# EVAL_DATASETS="--eval_livesports3k_cc --generate_method chat"

# --- 场景 5: 批量多数据集评测（同时评测多个数据集） ---
# EVAL_DATASETS="--eval_gigaspeech_test --eval_aishell1_test --eval_librispeech_test_clean"

# =============================================================================
#                        以下内容一般无需修改
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

OPT_ARGS=""
OPT_ARGS+=" --model_path ${MODEL_PATH}"
OPT_ARGS+=" --model_type ${MODEL_TYPE}"
OPT_ARGS+=" --model_name ${MODEL_NAME}"
OPT_ARGS+=" --answer_path ${ANSWER_PATH}"
OPT_ARGS+=" --batchsize ${BATCH_SIZE}"

if [ -n "${PT_PATH}" ]; then
    OPT_ARGS+=" --pt_path ${PT_PATH}"
fi

if [ -n "${MAX_SAMPLES}" ]; then
    OPT_ARGS+=" --max_sample_num ${MAX_SAMPLES}"
fi

OPT_ARGS+=" ${EVAL_DATASETS}"

MASTER_PORT=${MASTER_PORT:-29500}

CMD="torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} eval_main.py ${OPT_ARGS}"

echo "========================================"
echo "OmniEvalKit Evaluation"
echo "========================================"
echo "Model:    ${MODEL_PATH}"
echo "Type:     ${MODEL_TYPE}"
echo "GPUs:     ${GPUS_PER_NODE}"
echo "Batch:    ${BATCH_SIZE}"
echo "Output:   ${ANSWER_PATH}"
echo "Datasets: ${EVAL_DATASETS}"
echo "========================================"
echo ""
echo "CMD: ${CMD}"
echo ""

${CMD}
