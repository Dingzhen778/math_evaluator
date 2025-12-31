#!/usr/bin/env python3
"""
多GPU并行补充评测脚本
策略：将缺失任务按模型分组，每个GPU运行不同的模型
"""

import subprocess
import os
import glob
from collections import defaultdict
from datetime import datetime
import time

PYTHON = "/volume/data/rhjiang/miniconda3/envs/math/bin/python"
MODEL_BASE = "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_DIR = "/volume/data/rhjiang/math_evaluator/results-all-1227"
EVAL_SCRIPT = "/volume/data/rhjiang/math_evaluator/run_eval.py"

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

DATASETS = ["math500", "gsm8k", "aime2024", "aime2025", "gaokao2023en", "mathodyssey", "amc23", "olympiadbench_oe"]


def get_model_format(model):
    if model in LLAMA_MODELS:
        return "hash"
    return "boxed"


def count_existing_evals():
    model_dataset_counts = defaultdict(int)
    all_models = LLAMA_MODELS + QWEN_MODELS

    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        fname = os.path.basename(f)
        for ds in DATASETS:
            if fname.startswith(ds + "_"):
                for model in all_models:
                    if model in fname:
                        model_dataset_counts[(model, ds)] += 1
                        break
                break

    return model_dataset_counts


def get_models_needing_work():
    """获取需要补跑的模型及其缺失的数据集轮次"""
    all_models = LLAMA_MODELS + QWEN_MODELS
    model_dataset_counts = count_existing_evals()

    model_tasks = defaultdict(list)
    for model in all_models:
        for ds in DATASETS:
            count = model_dataset_counts[(model, ds)]
            need = 5 - count
            for _ in range(need):
                model_tasks[model].append(ds)

    # 按任务数排序
    return sorted(model_tasks.items(), key=lambda x: -len(x[1]))


def main():
    models_needing_work = get_models_needing_work()
    models_needing_work = [(m, tasks) for m, tasks in models_needing_work if tasks]

    total_tasks = sum(len(tasks) for _, tasks in models_needing_work)

    print(f"=" * 60)
    print(f"多GPU并行补充评测")
    print(f"=" * 60)
    print(f"需要补跑的模型数: {len(models_needing_work)}")
    print(f"总任务数: {total_tasks}")
    print(f"可用GPU: {AVAILABLE_GPUS}")
    print(f"=" * 60)

    if not models_needing_work:
        print("无需补跑，所有评测已完成！")
        return

    model_idx = 0
    batch_num = 0
    completed = 0
    failed = 0

    while model_idx < len(models_needing_work):
        batch_num += 1
        batch_size = min(len(AVAILABLE_GPUS), len(models_needing_work) - model_idx)

        print(f"\n=== 批次 {batch_num}: 模型 {model_idx+1}-{model_idx+batch_size}/{len(models_needing_work)} ===")

        processes = []
        for i in range(batch_size):
            model, tasks = models_needing_work[model_idx + i]
            gpu = AVAILABLE_GPUS[i]
            fmt = get_model_format(model)

            # 创建子进程脚本
            tasks_str = " ".join(tasks)
            cmd = f'''
cd /volume/data/rhjiang/math_evaluator
export CUDA_VISIBLE_DEVICES={gpu}
for ds in {tasks_str}; do
    echo "[$(date +%H:%M:%S)] GPU {gpu}: {model} - $ds"
    {PYTHON} {EVAL_SCRIPT} --model-path {MODEL_BASE}/{model} --dataset $ds --gpu 0 --results-dir {RESULTS_DIR} --answer-format {fmt} > {RESULTS_DIR}/log_missing_{model}_$ds.txt 2>&1
done
echo "[$(date +%H:%M:%S)] GPU {gpu}: {model} 完成 ({len(tasks)}个任务)"
'''
            p = subprocess.Popen(['bash', '-c', cmd])
            processes.append((p, model, gpu, len(tasks)))
            print(f"  启动: GPU {gpu} -> {model} ({len(tasks)}个任务)")

        # 等待当前批次完成
        for p, model, gpu, task_count in processes:
            p.wait()
            if p.returncode == 0:
                completed += task_count
            else:
                failed += task_count
            print(f"  完成: GPU {gpu} -> {model}")

        model_idx += batch_size
        print(f"  进度: {model_idx}/{len(models_needing_work)} 模型")

    print(f"\n{'=' * 60}")
    print(f"补充评测完成！")
    print(f"预计成功: ~{completed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
