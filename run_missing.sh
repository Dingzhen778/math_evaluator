#!/bin/bash
#===============================================================================
# 补跑遗漏的评测（跑完后删除此脚本）
#===============================================================================

PYTHON="/root/miniconda3/envs/math/bin/python"
BASE_PATH="/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_BASE="/volume/data/rhjiang/math_evaluator/results-1224"

cd /volume/data/rhjiang/math_evaluator

echo "=== 补跑遗漏的评测 ==="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 定义缺失的任务: round|model|dataset
MISSING_TASKS=(
    "1|Qwen3-4B-numinamath-reliable-random-32k-1e|olympiadbench_oe"
    "2|Qwen3-4B-numinamath-reliable-random-32k-1e|olympiadbench_oe"
    "3|Qwen3-4B-numinamath-reliable-random-32k-1e|olympiadbench_oe"
    "3|Qwen3-4B-numinamath-reliable-random-32k-3e|olympiadbench_oe"
)

# 使用4个GPU并行跑4个任务
run_single() {
    local GPU=$1
    local ROUND=$2
    local MODEL=$3
    local DATASET=$4
    local ROUND_DIR="${RESULTS_BASE}/round${ROUND}"

    echo "[$(date '+%H:%M:%S')] GPU$GPU: Round$ROUND $MODEL $DATASET"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON run_eval.py \
        --model-path "${BASE_PATH}/${MODEL}" \
        --gpu 0 \
        --dataset "$DATASET" \
        --results-dir "$ROUND_DIR" 2>&1 | tail -5

    echo "[$(date '+%H:%M:%S')] GPU$GPU: 完成"
}

# 并行运行4个任务
GPU=0
for task in "${MISSING_TASKS[@]}"; do
    ROUND=$(echo "$task" | cut -d'|' -f1)
    MODEL=$(echo "$task" | cut -d'|' -f2)
    DATASET=$(echo "$task" | cut -d'|' -f3)

    run_single $GPU "$ROUND" "$MODEL" "$DATASET" &
    GPU=$((GPU+1))
done

# 等待所有任务完成
wait

echo ""
echo "=== 补跑完成 ==="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 验证结果
echo ""
echo "=== 验证结果 ==="
for task in "${MISSING_TASKS[@]}"; do
    ROUND=$(echo "$task" | cut -d'|' -f1)
    MODEL=$(echo "$task" | cut -d'|' -f2)
    DATASET=$(echo "$task" | cut -d'|' -f3)

    if ls "${RESULTS_BASE}/round${ROUND}/${DATASET}_${MODEL}_"*.json &>/dev/null; then
        echo "✓ Round$ROUND $MODEL $DATASET"
    else
        echo "✗ Round$ROUND $MODEL $DATASET (失败)"
    fi
done

echo ""
echo "可以删除此脚本: rm /volume/data/rhjiang/math_evaluator/run_missing.sh"
