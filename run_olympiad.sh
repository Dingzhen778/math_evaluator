#!/bin/bash
# 补全 olympiadbench_oe

PYTHON=/volume/data/rhjiang/miniconda3/envs/math/bin/python
RESULTS_DIR=/volume/data/rhjiang/math_evaluator/results-1223/llama
MODEL_BASE=/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts

cd /volume/data/rhjiang/math_evaluator

# 第一批: 7个模型并行 (GPU 0,1,2,3,5,6,7)
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-1e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-2e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-3e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-1e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-2e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-3e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-1e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_7.txt 2>&1 &

echo "第一批7个已启动..."
wait

# 第二批: 5个模型
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-2e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-3e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-1e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-2e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_11.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-3e --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash > $RESULTS_DIR/log_olympiad_12.txt 2>&1 &

echo "第二批5个已启动..."
wait

echo "=== olympiadbench_oe 评测完成 ==="
