#!/bin/bash
#===============================================================================
# 补跑剩余评测: Round 2 (2e, 3e) + Round 3 (1e, 2e, 3e)
# 8张H200 GPU并行
#===============================================================================

source /root/miniconda3/etc/profile.d/conda.sh
conda activate math

cd /volume/data/rhjiang/math_evaluator

BASE_PATH="/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_BASE="/volume/data/rhjiang/math_evaluator/results-1224"

MODELS=(
    "Qwen3-4B-metamathqa-all-32k-1e"
    "Qwen3-4B-metamathqa-all-32k-2e"
    "Qwen3-4B-metamathqa-all-32k-3e"
)

DATASETS=(
    "math500"
    "gsm8k"
    "aime2024"
    "aime2025"
    "gaokao2023en"
    "mathodyssey"
    "amc23"
    "olympiadbench_oe"
)

NUM_GPUS=8

# 生成任务列表: (round, model, dataset)
declare -a TASKS=()

# Round 2: 2e 和 3e
for MODEL in "Qwen3-4B-metamathqa-all-32k-2e" "Qwen3-4B-metamathqa-all-32k-3e"; do
    for DS in "${DATASETS[@]}"; do
        TASKS+=("2|$MODEL|$DS")
    done
done

# Round 3: 全部模型
for MODEL in "${MODELS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        TASKS+=("3|$MODEL|$DS")
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "=========================================="
echo "剩余评测任务总数: $TOTAL_TASKS"
echo "使用 GPU 数量: $NUM_GPUS"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 创建必要目录
mkdir -p "${RESULTS_BASE}/round2"
mkdir -p "${RESULTS_BASE}/round3"

# 运行单个评测任务
run_task() {
    local GPU=$1
    local ROUND=$2
    local MODEL=$3
    local DATASET=$4
    local ROUND_DIR="${RESULTS_BASE}/round${ROUND}"
    local LOG_FILE="${ROUND_DIR}/log_${MODEL}_gpu${GPU}.txt"

    # 检查是否已完成
    if ls "${ROUND_DIR}"/${DATASET}_${MODEL}_*.json &>/dev/null; then
        echo "[$(date '+%H:%M:%S')] GPU$GPU: Skip $DATASET $MODEL round$ROUND (exists)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] GPU$GPU: Start $DATASET $MODEL round$ROUND"

    CUDA_VISIBLE_DEVICES=$GPU python run_eval.py \
        --model-path "${BASE_PATH}/${MODEL}" \
        --gpu 0 \
        --dataset "$DATASET" \
        --results-dir "$ROUND_DIR" >> "$LOG_FILE" 2>&1

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GPU$GPU: Done $DATASET $MODEL round$ROUND"
    else
        echo "[$(date '+%H:%M:%S')] GPU$GPU: FAILED $DATASET $MODEL round$ROUND (exit=$STATUS)"
    fi
    return $STATUS
}

export -f run_task
export BASE_PATH RESULTS_BASE

# 任务调度
TASK_IDX=0
declare -a PIDS=()
declare -a GPU_TASK=()

for GPU in $(seq 0 $((NUM_GPUS-1))); do
    PIDS[$GPU]=""
    GPU_TASK[$GPU]=""
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
                GPU_TASK[$GPU]=""
            fi
        fi

        # 如果GPU空闲且有任务，分配新任务
        if [ -z "${PIDS[$GPU]}" ] && [ $TASK_IDX -lt $TOTAL_TASKS ]; then
            TASK="${TASKS[$TASK_IDX]}"
            ROUND=$(echo "$TASK" | cut -d'|' -f1)
            MODEL=$(echo "$TASK" | cut -d'|' -f2)
            DATASET=$(echo "$TASK" | cut -d'|' -f3)

            run_task $GPU "$ROUND" "$MODEL" "$DATASET" &
            PIDS[$GPU]=$!
            GPU_TASK[$GPU]="$DATASET"
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
    COMPLETED=$(find "${RESULTS_BASE}/round2" "${RESULTS_BASE}/round3" -name "*.json" 2>/dev/null | wc -l)

    # 显示每个GPU当前任务
    GPU_STATUS=""
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        if [ -n "${GPU_TASK[$GPU]}" ]; then
            GPU_STATUS="${GPU_STATUS}GPU${GPU}:${GPU_TASK[$GPU]} "
        fi
    done

    echo "[$(date '+%H:%M:%S')] 进度: $COMPLETED/$TOTAL_TASKS | 任务分配: $TASK_IDX/$TOTAL_TASKS | 耗时: $((ELAPSED/60))m$((ELAPSED%60))s | $GPU_STATUS"

    sleep 30
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
echo "=== 结果统计 ==="
echo "Round 2:"
for MODEL in "${MODELS[@]}"; do
    COUNT=$(ls "${RESULTS_BASE}/round2"/*_${MODEL}_*.json 2>/dev/null | wc -l)
    echo "  $MODEL: $COUNT/8"
done

echo "Round 3:"
for MODEL in "${MODELS[@]}"; do
    COUNT=$(ls "${RESULTS_BASE}/round3"/*_${MODEL}_*.json 2>/dev/null | wc -l)
    echo "  $MODEL: $COUNT/8"
done
