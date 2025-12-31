#!/bin/bash
# 检查缺失的评测

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

BASE="/volume/data/rhjiang/math_evaluator/results-1224"

echo "=== 缺失评测检查 ==="
echo ""

for r in 1 2 3; do
    echo "--- Round $r ---"
    missing=0
    for model in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            if ! ls "${BASE}/round${r}/${ds}_${model}_"*.json &>/dev/null; then
                echo "缺失: Round$r | $model | $ds"
                missing=$((missing+1))
            fi
        done
    done
    if [ $missing -eq 0 ]; then
        echo "Round$r: 全部完成 ✓"
    else
        echo "Round$r 缺失: $missing 个"
    fi
    echo ""
done
