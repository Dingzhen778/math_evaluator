#!/bin/bash
# 补全缺失的评测任务

PYTHON=/volume/data/rhjiang/miniconda3/envs/math/bin/python
RESULTS_DIR=/volume/data/rhjiang/math_evaluator/results-1223/llama
MODEL_BASE=/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts

cd /volume/data/rhjiang/math_evaluator

# 第一批: 7个模型并行 (GPU 0,1,2,3,5,6,7)
echo "=== 第一批评测开始 ==="

# GPU 0: metamathqa-method-reliable-1e
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-1e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-1e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-1e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-1e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_metamathqa-method-reliable-1e_补全.txt 2>&1 &

# GPU 1: metamathqa-method-reliable-2e
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-2e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-2e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-2e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-2e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_metamathqa-method-reliable-2e_补全.txt 2>&1 &

# GPU 2: metamathqa-method-reliable-3e
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-3e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-3e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-3e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-metamathqa-method-reliable-3e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_metamathqa-method-reliable-3e_补全.txt 2>&1 &

# GPU 3: MetaMathQA-all-1e
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-1e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-1e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-1e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-1e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_MetaMathQA-all-1e_补全.txt 2>&1 &

# GPU 5: MetaMathQA-all-2e
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-2e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-2e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-2e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-2e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_MetaMathQA-all-2e_补全.txt 2>&1 &

# GPU 6: MetaMathQA-all-3e
CUDA_VISIBLE_DEVICES=6 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-3e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=6 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-3e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=6 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-3e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=6 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-MetaMathQA-all-3e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_MetaMathQA-all-3e_补全.txt 2>&1 &

# GPU 7: numinamath-method-reliable-1e
CUDA_VISIBLE_DEVICES=7 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-1e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=7 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-1e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=7 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-1e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=7 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-1e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_numinamath-method-reliable-1e_补全.txt 2>&1 &

echo "第一批7个模型已启动，等待完成..."
wait

echo "=== 第二批评测开始 ==="

# GPU 0: numinamath-method-reliable-2e
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-2e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-2e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-2e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=0 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-2e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_numinamath-method-reliable-2e_补全.txt 2>&1 &

# GPU 1: numinamath-method-reliable-3e
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-3e \
  --dataset gaokao2023en --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-3e \
  --dataset mathodyssey --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-3e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=1 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-method-reliable-3e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_numinamath-method-reliable-3e_补全.txt 2>&1 &

# GPU 2: numinamath-all-1e (只需要olympiadbench_oe)
CUDA_VISIBLE_DEVICES=2 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-1e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_numinamath-all-1e_补全.txt 2>&1 &

# GPU 3: numinamath-all-2e (只需要olympiadbench_oe)
CUDA_VISIBLE_DEVICES=3 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-2e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_numinamath-all-2e_补全.txt 2>&1 &

# GPU 5: numinamath-all-3e (需要amc23和olympiadbench_oe)
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-3e \
  --dataset amc23 --gpu 0 --results-dir $RESULTS_DIR --answer-format hash && \
CUDA_VISIBLE_DEVICES=5 $PYTHON run_eval.py \
  --model-path $MODEL_BASE/llama3_1-8B-numinamath-all-3e \
  --dataset olympiadbench_oe --gpu 0 --results-dir $RESULTS_DIR --answer-format hash \
  > $RESULTS_DIR/log_numinamath-all-3e_补全.txt 2>&1 &

echo "第二批5个模型已启动，等待完成..."
wait

echo "=== 所有评测完成 ==="
