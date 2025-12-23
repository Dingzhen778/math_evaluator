#!/bin/bash
#===============================================================================
# 新数据集评测脚本: GaoKao2023EN, MathOdyssey, AMC23, OlympiadBench
# 4张GPU并行，5轮评测
#===============================================================================

MODELS=(
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-3e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-3e"
)

DATASETS="gaokao2023en,mathodyssey,amc23,olympiadbench_oe,olympiadbench_tp"
RESULTS_BASE="results-1222-new"
NUM_GPUS=4
NUM_ROUNDS=5

source /root/miniconda3/etc/profile.d/conda.sh
conda activate math

mkdir -p "$RESULTS_BASE"

echo "=========================================="
echo "新数据集评测: GaoKao2023EN, MathOdyssey, AMC23, OlympiadBench"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "模型: ${#MODELS[@]} 个 x $NUM_ROUNDS 轮 x 5数据集"
echo "=========================================="

# 生成任务列表
declare -a TASKS=()
for ROUND in $(seq 1 $NUM_ROUNDS); do
    for MODEL_PATH in "${MODELS[@]}"; do
        TASKS+=("$MODEL_PATH|$ROUND")
    done
done

TOTAL_TASKS=${#TASKS[@]}
START_TIME=$(date +%s)

run_one() {
    local GPU=$1 MODEL=$2 ROUND=$3
    local ROUND_DIR="${RESULTS_BASE}/round${ROUND}"
    local MODEL_NAME=$(basename "$MODEL")
    local LOG_FILE="${RESULTS_BASE}/gpu${GPU}.log"
    mkdir -p "$ROUND_DIR"

    # 检查是否已完成
    local DONE=0
    for DS in gaokao2023en mathodyssey amc23 olympiadbench_oe olympiadbench_tp; do
        ls "$ROUND_DIR"/${DS}_${MODEL_NAME}_*.json &>/dev/null && DONE=$((DONE+1))
    done
    [ $DONE -eq 5 ] && return 0

    echo "[$(date '+%H:%M:%S')] Round${ROUND} - ${MODEL_NAME}" >> "$LOG_FILE"

    # 逐个数据集评测
    for DS in gaokao2023en mathodyssey amc23 olympiadbench_oe olympiadbench_tp; do
        if ! ls "$ROUND_DIR"/${DS}_${MODEL_NAME}_*.json &>/dev/null; then
            python run_eval.py --dataset "$DS" --model-path "$MODEL" --gpu "$GPU" --results-dir "$ROUND_DIR" >> "$LOG_FILE" 2>&1
        fi
    done

    echo "[$(date '+%H:%M:%S')] Round${ROUND} - ${MODEL_NAME} 完成" >> "$LOG_FILE"
}

# 任务调度
TASK_IDX=0
declare -a PIDS=()

for GPU in $(seq 0 $((NUM_GPUS-1))); do PIDS[$GPU]=""; done

while true; do
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        if [ -n "${PIDS[$GPU]}" ]; then
            if ! kill -0 ${PIDS[$GPU]} 2>/dev/null; then
                wait ${PIDS[$GPU]} 2>/dev/null
                PIDS[$GPU]=""
            fi
        fi

        if [ -z "${PIDS[$GPU]}" ] && [ $TASK_IDX -lt $TOTAL_TASKS ]; then
            TASK="${TASKS[$TASK_IDX]}"
            MODEL=$(echo "$TASK" | cut -d'|' -f1)
            ROUND=$(echo "$TASK" | cut -d'|' -f2)
            MODEL_NAME=$(basename "$MODEL")

            echo "[$(date '+%H:%M:%S')] GPU$GPU 开始: Round$ROUND - $MODEL_NAME"
            run_one $GPU "$MODEL" "$ROUND" &
            PIDS[$GPU]=$!
            TASK_IDX=$((TASK_IDX+1))
        fi
    done

    ALL_DONE=true
    for GPU in $(seq 0 $((NUM_GPUS-1))); do
        [ -n "${PIDS[$GPU]}" ] && ALL_DONE=false && break
    done
    [ "$ALL_DONE" = true ] && [ $TASK_IDX -ge $TOTAL_TASKS ] && break

    # 显示进度
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    RESULT_COUNT=$(find "$RESULTS_BASE" -name "*.json" 2>/dev/null | wc -l)
    EXPECTED=$((${#MODELS[@]} * NUM_ROUNDS * 5))
    echo "[$(date '+%H:%M:%S')] 进度: $RESULT_COUNT/$EXPECTED | 已耗时: $((ELAPSED/60))分$((ELAPSED%60))秒"

    sleep 10
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "评测完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: $((DURATION/3600))小时$((DURATION%3600/60))分$((DURATION%60))秒"
echo "=========================================="

# 汇总结果
python3 << 'EOF'
import json
from collections import defaultdict
from pathlib import Path

results = defaultdict(lambda: defaultdict(list))
for f in Path("results-1222-new").glob("round*/*.json"):
    try:
        d = json.load(open(f))
        results[d['model']][d['dataset']].append(d['accuracy'])
    except: pass

print("=" * 60)
print("结果汇总 (平均准确率)")
print("=" * 60)
for m in sorted(results):
    print(f"\n{m}")
    for ds in ['gaokao2023en', 'mathodyssey', 'amc23', 'olympiadbench_oe', 'olympiadbench_tp']:
        if ds in results[m]:
            v = results[m][ds]
            print(f"  {ds:15s}: {sum(v)/len(v):5.2f}% ({len(v)}轮)")

json.dump({m: {d: round(sum(v)/len(v), 2) for d,v in ds_dict.items()}
           for m, ds_dict in results.items()},
          open("results-1222-new/summary.json", 'w'), indent=2)
print("\n汇总已保存: results-1222-new/summary.json")
EOF
