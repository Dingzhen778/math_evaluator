#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate math

MODELS=(
    "llama3_1-8B-MetaMathQA-all-1e"
    "llama3_1-8B-MetaMathQA-all-2e"
    "llama3_1-8B-MetaMathQA-all-3e"
    "llama3_1-8B-metamathqa-method-reliable-1e"
    "llama3_1-8B-metamathqa-method-reliable-2e"
    "llama3_1-8B-metamathqa-method-reliable-3e"
)

BASE_PATH="/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_DIR="results-1223/llama"
cd /volume/data/rhjiang/math_evaluator

for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Processing: $MODEL"
    echo "=========================================="

    for DS in gaokao2023en mathodyssey amc23 olympiadbench_oe; do
        if ls ${RESULTS_DIR}/${DS}_${MODEL}_*.json 1>/dev/null 2>&1; then
            echo "Skipping $DS (exists)"
            continue
        fi
        echo "Running $DS..."
        CUDA_VISIBLE_DEVICES=0 python run_eval.py \
            --model-path "${BASE_PATH}/${MODEL}" \
            --gpu 0 \
            --dataset $DS \
            --results-dir $RESULTS_DIR
    done
done
echo "GPU0 done!"
