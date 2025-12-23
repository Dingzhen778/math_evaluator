#!/bin/bash
# 并行批量评测脚本
# 使用 4 个 GPU 同时评测不同的任务，每个 GPU 运行一个模型实例

set -e

# 模型列表
MODELS=(
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-3e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-3e"
)

DATASETS=("math500" "gsm8k" "aime2024" "aime2025")
NUM_ROUNDS=5
NUM_GPUS=4
MAX_WORKERS=32

# 日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 任务队列文件
TASK_QUEUE="/tmp/eval_task_queue_$$.txt"
TASK_LOCK="/tmp/eval_task_lock_$$"

# 开始时间
START_TIME=$(date +%s)
echo "=========================================="
echo "并行批量评测开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "使用 $NUM_GPUS 个 GPU 并行评测"
echo "=========================================="
echo ""

# 生成所有待评测任务到文件
> "$TASK_QUEUE"
TOTAL_TASKS=0

for ROUND in $(seq 1 $NUM_ROUNDS); do
    ROUND_DIR="results/round${ROUND}"
    mkdir -p "$ROUND_DIR"

    for MODEL_PATH in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$MODEL_PATH")

        for DATASET in "${DATASETS[@]}"; do
            # 检查是否已完成
            if ls "$ROUND_DIR"/${DATASET}_${MODEL_NAME}_*.json 2>/dev/null | grep -q .; then
                echo "  [跳过] Round $ROUND - $MODEL_NAME - $DATASET (已完成)"
            else
                echo "$ROUND|$MODEL_PATH|$DATASET" >> "$TASK_QUEUE"
                TOTAL_TASKS=$((TOTAL_TASKS + 1))
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "待评测任务数: $TOTAL_TASKS"
echo "=========================================="
echo ""

if [ $TOTAL_TASKS -eq 0 ]; then
    echo "所有任务已完成！"
    rm -f "$TASK_QUEUE"
    exit 0
fi

# 获取并删除第一个任务（原子操作）
get_next_task() {
    (
        flock -x 200
        if [ -s "$TASK_QUEUE" ]; then
            head -1 "$TASK_QUEUE"
            tail -n +2 "$TASK_QUEUE" > "$TASK_QUEUE.tmp"
            mv "$TASK_QUEUE.tmp" "$TASK_QUEUE"
        fi
    ) 200>"$TASK_LOCK"
}

# GPU worker 函数
gpu_worker() {
    local GPU_ID=$1

    while true; do
        # 获取下一个任务
        TASK=$(get_next_task)

        if [ -z "$TASK" ]; then
            echo "[GPU $GPU_ID] 没有更多任务，退出"
            break
        fi

        IFS='|' read -r ROUND MODEL_PATH DATASET <<< "$TASK"
        MODEL_NAME=$(basename "$MODEL_PATH")
        ROUND_DIR="results/round${ROUND}"
        LOG_FILE="$LOG_DIR/gpu${GPU_ID}_round${ROUND}_${MODEL_NAME}_${DATASET}.log"

        echo "[GPU $GPU_ID] 开始: Round $ROUND - $MODEL_NAME - $DATASET"

        # 运行评测
        if python run_eval.py \
            --dataset "$DATASET" \
            --model-path "$MODEL_PATH" \
            --gpu "$GPU_ID" \
            --tensor-parallel-size 1 \
            --max-workers $MAX_WORKERS \
            --results-dir "$ROUND_DIR" \
            > "$LOG_FILE" 2>&1; then
            echo "[GPU $GPU_ID] 完成: Round $ROUND - $MODEL_NAME - $DATASET"
        else
            echo "[GPU $GPU_ID] 失败: Round $ROUND - $MODEL_NAME - $DATASET (日志: $LOG_FILE)"
        fi
    done
}

# 启动 4 个 GPU worker
echo "启动 $NUM_GPUS 个 GPU worker..."
echo ""

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_worker $GPU_ID &
done

# 等待所有 worker 完成
wait

# 清理临时文件
rm -f "$TASK_QUEUE" "$TASK_LOCK"

# 结束时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOUR=$((TOTAL_DURATION / 3600))
TOTAL_MIN=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo "并行批量评测完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${TOTAL_HOUR}小时${TOTAL_MIN}分${TOTAL_SEC}秒"
echo "=========================================="
echo ""

# 计算平均值
echo "=========================================="
echo "计算平均值..."
echo "=========================================="
echo ""

python3 << 'EOF'
import json
from collections import defaultdict
from pathlib import Path

results_base = Path("results")
rounds = [1, 2, 3, 4, 5]

all_results = defaultdict(lambda: defaultdict(list))

for round_num in rounds:
    round_dir = results_base / f"round{round_num}"
    if not round_dir.exists():
        continue

    for json_file in round_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model = data.get('model', 'unknown')
            dataset = data.get('dataset', 'unknown')
            accuracy = data.get('accuracy', 0)

            if accuracy > 0:
                all_results[model][dataset].append(accuracy)
        except Exception as e:
            print(f"读取文件 {json_file} 时出错: {e}")

print("=" * 80)
print("结果汇总（平均值）")
print("=" * 80)
print("")

for model in sorted(all_results.keys()):
    print(f"模型: {model}")
    print("-" * 80)

    for dataset in sorted(all_results[model].keys()):
        accuracies = all_results[model][dataset]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            std_dev = (sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
            print(f"  {dataset:12s}: {avg_accuracy:6.2f}% +/- {std_dev:5.2f}% ({len(accuracies)}轮)")
    print("")

summary_file = results_base / "summary_average.json"
summary_data = {}

for model in sorted(all_results.keys()):
    summary_data[model] = {}
    for dataset in sorted(all_results[model].keys()):
        accuracies = all_results[model][dataset]
        if accuracies:
            avg = sum(accuracies) / len(accuracies)
            summary_data[model][dataset] = {
                'average': round(avg, 2),
                'std_dev': round((sum((x - avg) ** 2 for x in accuracies) / len(accuracies)) ** 0.5, 2),
                'rounds': len(accuracies),
                'values': [round(x, 2) for x in accuracies]
            }

with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print(f"汇总结果已保存到: {summary_file}")
EOF

echo ""
echo "日志文件保存在: $LOG_DIR/"
