#!/bin/bash
#===============================================================================
# 全量评测脚本: 15个模型 x 3轮 x 8数据集
# 8张H200 GPU并行，每个GPU加载一个模型跑完8个数据集
#===============================================================================

# 使用 math 环境的 Python
PYTHON="/root/miniconda3/envs/math/bin/python"

cd /volume/data/rhjiang/math_evaluator

BASE_PATH="/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_BASE="/volume/data/rhjiang/math_evaluator/results-1224"

# 所有15个模型
MODELS=(
    "Qwen3-4B-metamathqa-all-32k-1e"
    "Qwen3-4B-metamathqa-all-32k-2e"
    "Qwen3-4B-metamathqa-all-32k-3e"
    "Qwen3-4B-metamathqa-reliable-32k-1e"
    "Qwen3-4B-metamathqa-reliable-32k-2e"
    "Qwen3-4B-metamathqa-reliable-32k-3e"
    "Qwen3-4B-numinamath-all-32k-1e"
    "Qwen3-4B-numinamath-all-32k-2e"
    "Qwen3-4B-numinamath-all-32k-3e"
    "Qwen3-4B-numinamath-reliable-32k-1e"
    "Qwen3-4B-numinamath-reliable-32k-2e"
    "Qwen3-4B-numinamath-reliable-32k-3e"
    "Qwen3-4B-numinamath-reliable-random-32k-1e"
    "Qwen3-4B-numinamath-reliable-random-32k-2e"
    "Qwen3-4B-numinamath-reliable-random-32k-3e"
)

NUM_GPUS=8

# 创建目录
mkdir -p "${RESULTS_BASE}/round1"
mkdir -p "${RESULTS_BASE}/round2"
mkdir -p "${RESULTS_BASE}/round3"

# 生成任务列表: (round, model)
# 每个任务会在单个GPU上加载模型一次，然后跑完8个数据集
declare -a TASKS=()

for ROUND in 1 2 3; do
    for MODEL in "${MODELS[@]}"; do
        ROUND_DIR="${RESULTS_BASE}/round${ROUND}"
        # 检查该模型在该轮是否已完成8个数据集
        DONE_COUNT=$(ls "${ROUND_DIR}"/*_${MODEL}_*.json 2>/dev/null | wc -l)
        if [ "$DONE_COUNT" -lt 8 ]; then
            TASKS+=("${ROUND}|${MODEL}")
        fi
    done
done

TOTAL_TASKS=${#TASKS[@]}

echo "=========================================="
echo "全量评测: 15模型 x 3轮 x 8数据集"
echo "待完成任务组数: $TOTAL_TASKS (每组=1模型x8数据集)"
echo "使用 GPU 数量: $NUM_GPUS"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

if [ $TOTAL_TASKS -eq 0 ]; then
    echo "所有评测已完成！"
    exit 0
fi

# 运行单个模型的全部数据集评测
run_model_eval() {
    local GPU=$1
    local ROUND=$2
    local MODEL=$3
    local ROUND_DIR="${RESULTS_BASE}/round${ROUND}"
    local LOG_FILE="${ROUND_DIR}/log_${MODEL}.txt"

    echo "[$(date '+%H:%M:%S')] GPU$GPU: 开始 Round$ROUND $MODEL (8数据集)"

    # 使用 --dataset all 一次性跑完8个数据集，模型只加载一次
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON run_eval.py \
        --model-path "${BASE_PATH}/${MODEL}" \
        --gpu 0 \
        --dataset all \
        --results-dir "$ROUND_DIR" >> "$LOG_FILE" 2>&1

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GPU$GPU: 完成 Round$ROUND $MODEL"
    else
        echo "[$(date '+%H:%M:%S')] GPU$GPU: 失败 Round$ROUND $MODEL (exit=$STATUS)"
    fi
    return $STATUS
}

# 任务调度
TASK_IDX=0
declare -a PIDS=()
declare -a GPU_INFO=()

for GPU in $(seq 0 $((NUM_GPUS-1))); do
    PIDS[$GPU]=""
    GPU_INFO[$GPU]=""
done

START_TIME=$(date +%s)

while true; do
    # 检查每个GPU状态
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        # 检查进程是否完成
        if [ -n "${PIDS[$GPU]}" ]; then
            if ! kill -0 ${PIDS[$GPU]} 2>/dev/null; then
                wait ${PIDS[$GPU]} 2>/dev/null
                PIDS[$GPU]=""
                GPU_INFO[$GPU]=""
            fi
        fi

        # 如果GPU空闲且有任务，分配新任务
        if [ -z "${PIDS[$GPU]}" ] && [ $TASK_IDX -lt $TOTAL_TASKS ]; then
            TASK="${TASKS[$TASK_IDX]}"
            ROUND=$(echo "$TASK" | cut -d'|' -f1)
            MODEL=$(echo "$TASK" | cut -d'|' -f2)

            run_model_eval $GPU "$ROUND" "$MODEL" &
            PIDS[$GPU]=$!
            GPU_INFO[$GPU]="R${ROUND}:${MODEL:0:20}"
            TASK_IDX=$((TASK_IDX+1))
        fi
    done

    # 检查是否全部完成
    ALL_DONE=true
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        if [ -n "${PIDS[$GPU]}" ]; then
            ALL_DONE=false
            break
        fi
    done

    if [ "$ALL_DONE" = true ] && [ $TASK_IDX -ge $TOTAL_TASKS ]; then
        break
    fi

    # 显示进度
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # 统计已完成的json文件数
    R1=$(ls "${RESULTS_BASE}/round1"/*.json 2>/dev/null | wc -l)
    R2=$(ls "${RESULTS_BASE}/round2"/*.json 2>/dev/null | wc -l)
    R3=$(ls "${RESULTS_BASE}/round3"/*.json 2>/dev/null | wc -l)
    TOTAL_JSON=$((R1 + R2 + R3))

    # 显示每个GPU状态
    echo -n "[$(date '+%H:%M:%S')] 进度: $TOTAL_JSON/360 json | 任务: $TASK_IDX/$TOTAL_TASKS | 耗时: $((ELAPSED/60))m | GPU: "
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        if [ -n "${GPU_INFO[$GPU]}" ]; then
            echo -n "$GPU:${GPU_INFO[$GPU]:0:15} "
        else
            echo -n "$GPU:idle "
        fi
    done
    echo ""

    sleep 60
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "评测完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: $((DURATION/3600))h$((DURATION%3600/60))m$((DURATION%60))s"
echo "=========================================="

# 统计结果
echo ""
echo "=== 最终统计 ==="
for ROUND in 1 2 3; do
    echo "Round $ROUND:"
    for MODEL in "${MODELS[@]}"; do
        COUNT=$(ls "${RESULTS_BASE}/round${ROUND}"/*_${MODEL}_*.json 2>/dev/null | wc -l)
        if [ "$COUNT" -eq 8 ]; then
            STATUS="✓"
        else
            STATUS="✗"
        fi
        echo "  $MODEL: $COUNT/8 $STATUS"
    done
done
