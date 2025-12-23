#!/usr/bin/env python3
"""
评测 Qwen2.5-Math-1.5B 基础模型在7个数据集上的表现
使用 math-verify 进行答案提取和验证
"""

import os
import json
from datetime import datetime
from evaluators import MathEvaluator
from evaluators.vllm_client import VLLMClient

MODEL_PATH = "/volume/data/models/qwen3/Qwen2.5-Math-1.5B"
RESULTS_DIR = "/volume/data/rhjiang/math_evaluator/results-1222-mathverify"
GPU_IDS = "0,1,2,3"  # 使用全部4个H200

DATASETS = [
    'math500',
    'gsm8k',
    'aime2024',
    'aime2025',
    'gaokao2023en',
    'olympiadbench_oe',
    'mathodyssey',
]

def main():
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型（只加载一次）
    print("=" * 80)
    print(f"正在加载模型: {MODEL_PATH}")
    print("=" * 80)

    vllm_client = VLLMClient(
        model_path=MODEL_PATH,
        gpu_ids=GPU_IDS,
        gpu_memory_utilization=0.95,
        max_model_len=32768,  # 充分利用显存
        trust_remote_code=True,
        tensor_parallel_size=4,  # 4卡并行
    )

    print("模型加载完成！\n")

    # 生成函数
    def model_generate_fn(system_prompt: str, user_prompt: str) -> str:
        return vllm_client.generate(system_prompt, user_prompt)

    # 汇总结果
    all_results = {}

    # 评测每个数据集
    for dataset_name in DATASETS:
        print("=" * 80)
        print(f"评测数据集: {dataset_name}")
        print("=" * 80)

        evaluator = MathEvaluator(
            dataset_name=dataset_name,
            model_generate_fn=model_generate_fn,
            max_workers=128,  # 高并发
            use_few_shot=True,
            vllm_client=vllm_client,
        )

        # 加载数据集
        samples = evaluator.load_dataset()
        print(f"加载了 {len(samples)} 个样本")

        # 评测 (大batch加速)
        results = evaluator.evaluate(samples, model_generate_fn=model_generate_fn, batch_size=1024)

        # 保存结果
        model_name = os.path.basename(MODEL_PATH)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{dataset_name}_{model_name}_{timestamp}.json"
        output_path = os.path.join(RESULTS_DIR, output_filename)

        results['model'] = model_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n数据集: {dataset_name}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"正确数: {results['correct']}/{results['total']}")
        print(f"结果已保存到: {output_path}")
        print()

        all_results[dataset_name] = {
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total']
        }

    # 打印汇总
    print("\n" + "=" * 80)
    print("评测结果汇总: Qwen2.5-Math-1.5B")
    print("=" * 80)
    print(f"{'Dataset':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 50)
    for ds, r in all_results.items():
        print(f"{ds:<20} {r['accuracy']:>9.2f}% {r['correct']:>10} {r['total']:>10}")
    print("=" * 80)

    # 保存汇总
    summary_path = os.path.join(RESULTS_DIR, "qwen25_math_1.5b_base_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n汇总已保存到: {summary_path}")

if __name__ == "__main__":
    main()
