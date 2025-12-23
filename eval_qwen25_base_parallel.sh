#!/bin/bash
# 4张卡并行评测 Qwen2.5-Math-1.5B

MODEL_PATH="/volume/data/models/qwen3/Qwen2.5-Math-1.5B"
RESULTS_DIR="/volume/data/rhjiang/math_evaluator/results-1222-mathverify"

mkdir -p $RESULTS_DIR

# GPU 0: math500 + aime2024
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset math500 \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu0_math500.txt &
PID0=$!

# GPU 1: gsm8k
CUDA_VISIBLE_DEVICES=1 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset gsm8k \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu1_gsm8k.txt &
PID1=$!

# GPU 2: gaokao2023en + mathodyssey
CUDA_VISIBLE_DEVICES=2 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset gaokao2023en \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu2_gaokao.txt &
PID2=$!

# GPU 3: olympiadbench_oe + aime2025
CUDA_VISIBLE_DEVICES=3 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset olympiadbench_oe \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu3_olympiad.txt &
PID3=$!

echo "Started 4 parallel evaluations:"
echo "  GPU 0 (PID $PID0): math500"
echo "  GPU 1 (PID $PID1): gsm8k"
echo "  GPU 2 (PID $PID2): gaokao2023en"
echo "  GPU 3 (PID $PID3): olympiadbench_oe"

# 等待第一轮完成
wait $PID0
echo "GPU 0 完成 math500，开始 aime2024..."
CUDA_VISIBLE_DEVICES=0 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset aime2024 \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu0_aime2024.txt &
PID0=$!

wait $PID2
echo "GPU 2 完成 gaokao2023en，开始 mathodyssey..."
CUDA_VISIBLE_DEVICES=2 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset mathodyssey \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu2_odyssey.txt &
PID2=$!

wait $PID3
echo "GPU 3 完成 olympiadbench_oe，开始 aime2025..."
CUDA_VISIBLE_DEVICES=3 python run_eval.py \
    --model-path $MODEL_PATH \
    --dataset aime2025 \
    --gpu 0 \
    --results-dir $RESULTS_DIR 2>&1 | tee ${RESULTS_DIR}/log_gpu3_aime2025.txt &
PID3=$!

# 等待所有完成
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "=========================================="
echo "所有评测完成！"
echo "=========================================="
echo "结果保存在: $RESULTS_DIR"
ls -la $RESULTS_DIR/*.json
