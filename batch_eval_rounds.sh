#!/bin/bash
# 多轮批量评测脚本
# 运行5轮实验，结果分别存储到 round1, round2, round3, round4, round5 目录
# round1已完成，从round2开始运行4轮（round2, round3, round4, round5）
# 最后计算平均值

set +e  # 允许错误，以便可以跳过已完成的模型

# 模型列表（包含所有6个模型）
MODELS=(
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-reliable-32k-3e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts/Qwen2.5-math-1.5B-numinamath-default-32k-3e"
)

# 配置
GPU_IDS="0,1,2,3"  # 使用所有4个GPU
MAX_WORKERS=32
DATASETS="all"
NUM_ROUNDS=5  # 运行5轮（round1已完成，再跑4轮）

# 创建结果目录
mkdir -p results

# 开始时间
START_TIME=$(date +%s)
echo "=========================================="
echo "多轮批量评测开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "将运行 $NUM_ROUNDS 轮实验"
echo "=========================================="
echo ""

# 运行多轮实验（从round2开始，因为round1已完成）
for ROUND in $(seq 1 $NUM_ROUNDS); do
    ROUND_DIR="results/round${ROUND}"
    mkdir -p "$ROUND_DIR"
    
    echo ""
    echo "################################################################################"
    echo "# 第 $ROUND 轮实验开始"
    echo "# 结果将保存到: $ROUND_DIR"
    echo "################################################################################"
    echo ""
    
    ROUND_START=$(date +%s)
    
    # 遍历每个模型
    TOTAL=${#MODELS[@]}
    CURRENT=0
    
    for MODEL_PATH in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        MODEL_NAME=$(basename "$MODEL_PATH")
        
        echo ""
        echo "  ================================================================================"
        echo "  # 第 $ROUND 轮 - 评测进度: $CURRENT/$TOTAL"
        echo "  # 模型: $MODEL_NAME"
        echo "  ================================================================================"
        echo ""
        
        # 检查是否已经评测完成（检查是否有所有数据集的结果文件）
        SKIP_MODEL=false
        if [ "$DATASETS" = "all" ]; then
            # 检查是否有所有4个数据集的结果
            REQUIRED_DATASETS=("math500" "gsm8k" "aime2024" "aime2025")
            COMPLETED_COUNT=0
            for DS in "${REQUIRED_DATASETS[@]}"; do
                if ls "$ROUND_DIR"/${DS}_${MODEL_NAME}_*.json 2>/dev/null | grep -q .; then
                    COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                fi
            done
            
            if [ $COMPLETED_COUNT -eq ${#REQUIRED_DATASETS[@]} ]; then
                echo "  ✓ 模型 $MODEL_NAME 在第 $ROUND 轮已完成评测（找到 $COMPLETED_COUNT 个数据集的结果文件）"
                echo "    跳过此模型..."
                SKIP_MODEL=true
            fi
        else
            # 检查指定数据集是否已完成
            if ls "$ROUND_DIR"/${DATASETS}_${MODEL_NAME}_*.json 2>/dev/null | grep -q .; then
                echo "  ✓ 模型 $MODEL_NAME 的数据集 $DATASETS 在第 $ROUND 轮已完成评测"
                echo "    跳过此模型..."
                SKIP_MODEL=true
            fi
        fi
        
        # 如果已完成，跳过
        if [ "$SKIP_MODEL" = true ]; then
            echo ""
            continue
        fi
        
        # 记录开始时间
        MODEL_START=$(date +%s)
        
        # 运行评测
        if python batch_eval.py \
            --model-path "$MODEL_PATH" \
            --gpu "$GPU_IDS" \
            --max-workers $MAX_WORKERS \
            --datasets "$DATASETS" \
            --results-dir "$ROUND_DIR"; then
            
            MODEL_END=$(date +%s)
            MODEL_DURATION=$((MODEL_END - MODEL_START))
            MODEL_MIN=$((MODEL_DURATION / 60))
            MODEL_SEC=$((MODEL_DURATION % 60))
            
            echo ""
            echo "  ✓ 模型 $MODEL_NAME 第 $ROUND 轮评测完成 (耗时: ${MODEL_MIN}分${MODEL_SEC}秒)"
            echo ""
        else
            MODEL_END=$(date +%s)
            MODEL_DURATION=$((MODEL_END - MODEL_START))
            MODEL_MIN=$((MODEL_DURATION / 60))
            MODEL_SEC=$((MODEL_DURATION % 60))
            
            echo ""
            echo "  ✗ 模型 $MODEL_NAME 第 $ROUND 轮评测失败 (耗时: ${MODEL_MIN}分${MODEL_SEC}秒)"
            echo ""
        fi
    done
    
    ROUND_END=$(date +%s)
    ROUND_DURATION=$((ROUND_END - ROUND_START))
    ROUND_HOUR=$((ROUND_DURATION / 3600))
    ROUND_MIN=$(((ROUND_DURATION % 3600) / 60))
    ROUND_SEC=$((ROUND_DURATION % 60))
    
    echo ""
    echo "################################################################################"
    echo "# 第 $ROUND 轮实验完成 (耗时: ${ROUND_HOUR}小时${ROUND_MIN}分${ROUND_SEC}秒)"
    echo "################################################################################"
    echo ""
done

# 结束时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOUR=$((TOTAL_DURATION / 3600))
TOTAL_MIN=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo "多轮批量评测完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
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
import os
from collections import defaultdict
from pathlib import Path

# 结果目录
results_base = Path("results-1219")
rounds = [1, 2, 3, 4, 5]  # 总共5轮

# 收集所有结果
all_results = defaultdict(lambda: defaultdict(list))  # model -> dataset -> [accuracies]

for round_num in rounds:
    round_dir = results_base / f"round{round_num}"
    if not round_dir.exists():
        print(f"警告: {round_dir} 不存在，跳过")
        continue
    
    # 遍历所有JSON文件
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

# 计算并显示平均值
print("=" * 80)
print("结果汇总（平均值）")
print("=" * 80)
print("")

# 按模型分组
for model in sorted(all_results.keys()):
    print(f"模型: {model}")
    print("-" * 80)
    
    for dataset in sorted(all_results[model].keys()):
        accuracies = all_results[model][dataset]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            std_dev = (sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            
            print(f"  {dataset:12s}: {avg_accuracy:6.2f}% ± {std_dev:5.2f}% "
                  f"(范围: {min_acc:.2f}% - {max_acc:.2f}%, {len(accuracies)}轮)")
        else:
            print(f"  {dataset:12s}: 无数据")
    
    print("")

# 保存汇总结果
summary_file = results_base / "summary_average.json"
summary_data = {}

for model in sorted(all_results.keys()):
    summary_data[model] = {}
    for dataset in sorted(all_results[model].keys()):
        accuracies = all_results[model][dataset]
        if accuracies:
            summary_data[model][dataset] = {
                'average': round(sum(accuracies) / len(accuracies), 2),
                'std_dev': round((sum((x - sum(accuracies)/len(accuracies)) ** 2 for x in accuracies) / len(accuracies)) ** 0.5, 2),
                'min': round(min(accuracies), 2),
                'max': round(max(accuracies), 2),
                'rounds': len(accuracies),
                'values': [round(x, 2) for x in accuracies]
            }

with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print("=" * 80)
print(f"汇总结果已保存到: {summary_file}")
print("=" * 80)
print("")
EOF

echo "=========================================="
echo "各轮结果目录:"
echo "=========================================="
for round_num in $(seq 1 $NUM_ROUNDS); do
    echo "Round $round_num: results/round${round_num}/"
    ls -lh "results/round${round_num}"/*.json 2>/dev/null | wc -l | xargs echo "  文件数:"
done
echo ""

