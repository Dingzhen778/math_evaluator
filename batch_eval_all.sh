#!/bin/bash
# 批量评测多个模型
# 使用方法: bash batch_eval_all.sh
# 功能: 自动跳过已完成的模型，只评测未完成的模型

# 注意：不要使用 set -e，因为我们需要处理跳过逻辑
set +e  # 允许错误，以便可以跳过已完成的模型

# 模型列表（包含所有6个模型）
MODELS=(
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-32k-3e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-reliable-32k-1e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-reliable-32k-2e"
    "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-reliable-32k-3e"
)

# 配置
GPU_IDS="0,1,2,3"  # 使用所有4个GPU
MAX_WORKERS=32
DATASETS="all"

# 创建结果目录
mkdir -p results

# 开始时间
START_TIME=$(date +%s)
echo "=========================================="
echo "批量评测开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 遍历每个模型
TOTAL=${#MODELS[@]}
CURRENT=0

for MODEL_PATH in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    echo ""
    echo "################################################################################"
    echo "# 评测进度: $CURRENT/$TOTAL"
    echo "# 模型: $MODEL_NAME"
    echo "# 路径: $MODEL_PATH"
    echo "################################################################################"
    echo ""
    
    # 检查是否已经评测完成（检查是否有所有数据集的结果文件）
    SKIP_MODEL=false
    if [ "$DATASETS" = "all" ]; then
        # 检查是否有所有4个数据集的结果
        REQUIRED_DATASETS=("math500" "gsm8k" "aime2024" "aime2025")
        COMPLETED_COUNT=0
        for DS in "${REQUIRED_DATASETS[@]}"; do
            if ls results/${DS}_${MODEL_NAME}_*.json 2>/dev/null | grep -q .; then
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            fi
        done
        
        if [ $COMPLETED_COUNT -eq ${#REQUIRED_DATASETS[@]} ]; then
            echo "✓ 模型 $MODEL_NAME 已完成评测（找到 $COMPLETED_COUNT 个数据集的结果文件）"
            echo "  跳过此模型..."
            SKIP_MODEL=true
        fi
    else
        # 检查指定数据集是否已完成
        if ls results/${DATASETS}_${MODEL_NAME}_*.json 2>/dev/null | grep -q .; then
            echo "✓ 模型 $MODEL_NAME 的数据集 $DATASETS 已完成评测"
            echo "  跳过此模型..."
            SKIP_MODEL=true
        fi
    fi
    
    # 如果已完成，跳过
    if [ "$SKIP_MODEL" = true ]; then
        echo ""
        REMAINING=$((TOTAL - CURRENT))
        if [ $REMAINING -gt 0 ]; then
            echo "剩余模型数: $REMAINING"
            echo ""
        fi
        continue
    fi
    
    # 记录开始时间
    MODEL_START=$(date +%s)
    
    # 运行评测
    if python batch_eval.py \
        --model-path "$MODEL_PATH" \
        --gpu "$GPU_IDS" \
        --max-workers $MAX_WORKERS \
        --datasets "$DATASETS"; then
        
        MODEL_END=$(date +%s)
        MODEL_DURATION=$((MODEL_END - MODEL_START))
        MODEL_MIN=$((MODEL_DURATION / 60))
        MODEL_SEC=$((MODEL_DURATION % 60))
        
        echo ""
        echo "✓ 模型 $MODEL_NAME 评测完成 (耗时: ${MODEL_MIN}分${MODEL_SEC}秒)"
        echo ""
    else
        MODEL_END=$(date +%s)
        MODEL_DURATION=$((MODEL_END - MODEL_START))
        MODEL_MIN=$((MODEL_DURATION / 60))
        MODEL_SEC=$((MODEL_DURATION % 60))
        
        echo ""
        echo "✗ 模型 $MODEL_NAME 评测失败 (耗时: ${MODEL_MIN}分${MODEL_SEC}秒)"
        echo ""
    fi
    
    # 显示进度
    REMAINING=$((TOTAL - CURRENT))
    if [ $REMAINING -gt 0 ]; then
        echo "剩余模型数: $REMAINING"
        echo ""
    fi
done

# 结束时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOUR=$((TOTAL_DURATION / 3600))
TOTAL_MIN=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo "批量评测完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${TOTAL_HOUR}小时${TOTAL_MIN}分${TOTAL_SEC}秒"
echo "=========================================="
echo ""
echo "结果文件保存在: results/"
echo ""
echo "=========================================="
echo "结果汇总（按模型分组）:"
echo "=========================================="
echo ""

# 按模型分组显示结果
for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "模型: $MODEL_NAME"
    
    # 查找该模型的所有结果文件
    RESULT_FILES=$(ls results/*${MODEL_NAME}*.json 2>/dev/null | sort)
    
    if [ -n "$RESULT_FILES" ]; then
        for RESULT_FILE in $RESULT_FILES; do
            # 提取数据集名称和准确率
            DATASET=$(basename "$RESULT_FILE" | cut -d'_' -f1)
            ACCURACY=$(python3 -c "import json; f=open('$RESULT_FILE'); d=json.load(f); print(f\"{d.get('accuracy', 'N/A')}% ({d.get('correct', 0)}/{d.get('total', 0)})\")" 2>/dev/null || echo "N/A")
            echo "  - $DATASET: $ACCURACY"
        done
    else
        echo "  - 未找到结果文件"
    fi
    echo ""
done

echo "=========================================="
echo "所有结果文件:"
echo "=========================================="
ls -lh results/*.json 2>/dev/null | awk '{print $9, $5}' || echo "未找到结果文件"
echo ""

