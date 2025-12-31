#!/usr/bin/env python3
"""
补充评测脚本: 补跑缺失的评测轮次
策略：逐个模型运行，避免GPU显存竞争
"""

import subprocess
import os
import glob
from collections import defaultdict
from datetime import datetime

PYTHON = "/volume/data/rhjiang/miniconda3/envs/math/bin/python"
MODEL_BASE = "/volume/data/lzh/Pai-Megatron-Patch-main/qwen-exp-ckpts"
RESULTS_DIR = "/volume/data/rhjiang/math_evaluator/results-all-1227"
EVAL_SCRIPT = "/volume/data/rhjiang/math_evaluator/run_eval.py"

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
    """获取模型对应的answer format"""
    if model in LLAMA_MODELS:
        return "hash"
    return "boxed"


def count_existing_evals():
    """统计已有的评测数量"""
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


def get_missing_tasks():
    """获取需要补跑的任务列表"""
    all_models = LLAMA_MODELS + QWEN_MODELS
    model_dataset_counts = count_existing_evals()

    missing = []
    for model in all_models:
        for ds in DATASETS:
            count = model_dataset_counts[(model, ds)]
            need = 5 - count
            if need > 0:
                missing.append((model, ds, need))

    return missing


def run_evaluation(model, dataset, round_num, gpu=0):
    """运行单个评测"""
    model_path = os.path.join(MODEL_BASE, model)
    fmt = get_model_format(model)
    log_file = os.path.join(RESULTS_DIR, f"log_missing_{model}_{dataset}_r{round_num}.txt")

    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--model-path", model_path,
        "--dataset", dataset,
        "--gpu", str(gpu),
        "--results-dir", RESULTS_DIR,
        "--answer-format", fmt,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 运行: {model} - {dataset} (第{round_num}轮)")

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        if result.returncode == 0:
            return True
        else:
            print(f"  !! 失败 (返回码: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"  !! 超时")
        return False
    except Exception as e:
        print(f"  !! 错误: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    args = parser.parse_args()

    missing = get_missing_tasks()

    # 统计总任务数
    total_tasks = sum(need for _, _, need in missing)

    print(f"=" * 60)
    print(f"补充评测")
    print(f"=" * 60)
    print(f"缺失模型+数据集组合: {len(missing)}")
    print(f"需要补跑的轮数: {total_tasks}")
    print(f"使用GPU: {args.gpu}")
    print(f"=" * 60)

    if args.dry_run:
        print("\n缺失列表:")
        for model, ds, need in sorted(missing, key=lambda x: -x[2])[:20]:
            print(f"  {model} + {ds}: 缺 {need} 轮")
        return

    # 按模型分组执行，减少模型加载次数
    model_tasks = defaultdict(list)
    for model, ds, need in missing:
        for _ in range(need):
            model_tasks[model].append(ds)

    completed = 0
    failed = 0

    for model, datasets_to_run in model_tasks.items():
        print(f"\n--- 模型: {model} ({len(datasets_to_run)}个任务) ---")

        # 对于同一个模型，用--dataset all会更高效
        # 但这里我们需要精确控制轮数，所以逐个数据集运行
        for i, ds in enumerate(datasets_to_run):
            success = run_evaluation(model, ds, i+1, args.gpu)
            if success:
                completed += 1
            else:
                failed += 1
            print(f"  进度: {completed + failed}/{total_tasks} (成功: {completed}, 失败: {failed})")

    print(f"\n{'=' * 60}")
    print(f"补充评测完成！")
    print(f"成功: {completed}")
    print(f"失败: {failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
