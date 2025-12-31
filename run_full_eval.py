#!/usr/bin/env python3
"""
全量评测脚本: 34个模型 × 8数据集 × 5轮 = 1360组评测

策略：每个GPU同时只跑一个模型，7个GPU并行跑7个不同的模型
"""

import subprocess
import os
import time
from pathlib import Path
from datetime import datetime

PYTHON = "/volume/data/rhjiang/miniconda3/envs/math/bin/python"
MODEL_BASE = "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_DIR = "/volume/data/rhjiang/math_evaluator/results-all-1227"
EVAL_SCRIPT = "/volume/data/rhjiang/math_evaluator/run_eval.py"

# 可用GPU (排除GPU 4)
AVAILABLE_GPUS = [0, 1, 2, 3, 5, 6, 7]

# LLaMA 模型 (使用 hash 格式)
LLAMA_MODELS = [
    "llama3_1-8B-MetaMathQA-all-1e",
    "llama3_1-8B-MetaMathQA-all-2e",
    "llama3_1-8B-MetaMathQA-all-3e",
    "llama3_1-8B-metamathqa-method-reliable-1e",
    "llama3_1-8B-metamathqa-method-reliable-2e",
    "llama3_1-8B-metamathqa-method-reliable-3e",
    "llama3_1-8B-metamathqa-nogreedy-reliable-1e",
    "llama3_1-8B-metamathqa-nogreedy-reliable-2e",
    "llama3_1-8B-metamathqa-nogreedy-reliable-3e",
    "llama3_1-8B-metamathqa-random-reliable-1e",
    "llama3_1-8B-metamathqa-random-reliable-2e",
    "llama3_1-8B-metamathqa-random-reliable-3e",
    "llama3_1-8B-numinamath-all-1e",
    "llama3_1-8B-numinamath-all-2e",
    "llama3_1-8B-numinamath-all-3e",
    "llama3_1-8B-numinamath-method-reliable-1e",
    "llama3_1-8B-numinamath-method-reliable-2e",
    "llama3_1-8B-numinamath-method-reliable-3e",
]

# Qwen 模型 (使用 boxed 格式)
QWEN_MODELS = [
    "Qwen2.5-math-1.5B-metamathqa-all-32k-1e",
    "Qwen2.5-math-1.5B-metamathqa-all-32k-2e",
    "Qwen2.5-math-1.5B-metamathqa-all-32k-3e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-32k-1e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-32k-2e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-32k-3e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-nogreedy-32k-1e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-random-32k-1e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-random-32k-2e",
    "Qwen2.5-math-1.5B-metamathqa-reliable-random-32k-3e",
    "Qwen2.5-math-1.5B-numinamath-default-32k-1e",
    "Qwen2.5-math-1.5B-numinamath-default-32k-2e",
    "Qwen2.5-math-1.5B-numinamath-default-32k-3e",
    "Qwen2.5-math-1.5B-numinamath-reliable-32k-1e",
    "Qwen2.5-math-1.5B-numinamath-reliable-32k-2e",
    "Qwen2.5-math-1.5B-numinamath-reliable-32k-3e",
]

NUM_ROUNDS = 5

def run_model_all_rounds(model, answer_format, gpu):
    """在指定GPU上跑一个模型的5轮评测"""
    model_path = os.path.join(MODEL_BASE, model)

    for round_num in range(1, NUM_ROUNDS + 1):
        log_file = os.path.join(RESULTS_DIR, f"log_{model}_round{round_num}.txt")

        cmd = [
            PYTHON, EVAL_SCRIPT,
            "--model-path", model_path,
            "--dataset", "all",
            "--gpu", "0",
            "--results-dir", RESULTS_DIR,
            "--answer-format", answer_format,
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu}: {model} 第{round_num}/5轮")

        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode != 0:
                print(f"  !! {model} 第{round_num}轮失败")
        except subprocess.TimeoutExpired:
            print(f"  !! {model} 第{round_num}轮超时")
        except Exception as e:
            print(f"  !! {model} 第{round_num}轮错误: {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu}: {model} 完成5轮")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 生成任务列表：每个任务是(模型, 格式)
    all_models = [(m, "hash") for m in LLAMA_MODELS] + [(m, "boxed") for m in QWEN_MODELS]

    total_models = len(all_models)
    print(f"=" * 60)
    print(f"全量评测启动")
    print(f"=" * 60)
    print(f"模型总数: {total_models}")
    print(f"每模型轮数: {NUM_ROUNDS}")
    print(f"总评测组数: {total_models * 8 * NUM_ROUNDS}")
    print(f"可用GPU: {AVAILABLE_GPUS}")
    print(f"=" * 60)

    # 分批执行，每批使用所有可用GPU
    model_idx = 0
    batch_num = 0

    while model_idx < total_models:
        batch_num += 1
        batch_size = min(len(AVAILABLE_GPUS), total_models - model_idx)

        print(f"\n=== 批次 {batch_num}: 模型 {model_idx+1}-{model_idx+batch_size}/{total_models} ===")

        # 启动当前批次的进程
        processes = []
        for i in range(batch_size):
            model, fmt = all_models[model_idx + i]
            gpu = AVAILABLE_GPUS[i]

            # 使用subprocess启动独立进程
            cmd = f'''
cd /volume/data/rhjiang/math_evaluator
export CUDA_VISIBLE_DEVICES={gpu}
for r in 1 2 3 4 5; do
    echo "[$(date +%H:%M:%S)] GPU {gpu}: {model} 第$r/5轮"
    {PYTHON} {EVAL_SCRIPT} --model-path {MODEL_BASE}/{model} --dataset all --gpu 0 --results-dir {RESULTS_DIR} --answer-format {fmt} > {RESULTS_DIR}/log_{model}_round$r.txt 2>&1
done
echo "[$(date +%H:%M:%S)] GPU {gpu}: {model} 完成5轮"
'''
            p = subprocess.Popen(['bash', '-c', cmd])
            processes.append((p, model, gpu))
            print(f"  启动: GPU {gpu} -> {model}")

        # 等待当前批次完成
        for p, model, gpu in processes:
            p.wait()
            print(f"  完成: GPU {gpu} -> {model}")

        model_idx += batch_size

    print(f"\n{'=' * 60}")
    print(f"全量评测完成！")
    print(f"结果目录: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
