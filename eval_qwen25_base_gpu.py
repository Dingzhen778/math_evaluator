#!/usr/bin/env python3
"""
单GPU评测脚本 - 一次加载模型，连续评测多个数据集
用法: CUDA_VISIBLE_DEVICES=0 python eval_qwen25_base_gpu.py --datasets math500,aime2024
"""

import os
import sys
import json
import argparse
from datetime import datetime
from evaluators import MathEvaluator

MODEL_PATH = "/volume/data/models/qwen3/Qwen2.5-Math-1.5B-Instruct"
RESULTS_DIR = "/volume/data/rhjiang/math_evaluator/results-1222-mathverify"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='逗号分隔的数据集列表')
    parser.add_argument('--gpu-id', type=str, default='0', help='GPU ID (用于日志)')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 加载模型（只加载一次）
    print("=" * 80)
    print(f"[GPU {args.gpu_id}] 正在加载模型: {MODEL_PATH}")
    print(f"[GPU {args.gpu_id}] 待评测数据集: {datasets}")
    print("=" * 80)

    # 不设置 gpu_ids，让 VLLMClient 使用已有的 CUDA_VISIBLE_DEVICES
    # 需要修改 VLLMClient 不覆盖环境变量
    from vllm import LLM, SamplingParams
    import threading

    print(f"[GPU {args.gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_batched_tokens=32768,
        max_num_seqs=1024,
        enable_chunked_prefill=True,
    )

    # 创建简单的 client wrapper
    class SimpleVLLMClient:
        def __init__(self, llm):
            self.llm = llm
            self._lock = threading.Lock()

        def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
            if system_prompt:
                prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                prompt = user_prompt

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["\n\nQuestion:", "Question:"]
            )

            with self._lock:
                outputs = self.llm.generate([prompt], sampling_params)
                return outputs[0].outputs[0].text.strip()

        def generate_batch(self, prompts, temperature=0.0, max_tokens=2048):
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["\n\nQuestion:", "Question:"]
            )

            with self._lock:
                outputs = self.llm.generate(prompts, sampling_params)
                return [o.outputs[0].text.strip() for o in outputs]

    vllm_client = SimpleVLLMClient(llm)

    print(f"[GPU {args.gpu_id}] 模型加载完成！\n")

    def model_generate_fn(system_prompt: str, user_prompt: str) -> str:
        return vllm_client.generate(system_prompt, user_prompt)

    # 评测每个数据集
    for dataset_name in datasets:
        print("=" * 80)
        print(f"[GPU {args.gpu_id}] 评测数据集: {dataset_name}")
        print("=" * 80)

        evaluator = MathEvaluator(
            dataset_name=dataset_name,
            model_generate_fn=model_generate_fn,
            max_workers=64,
            use_few_shot=True,
            vllm_client=vllm_client,
        )

        samples = evaluator.load_dataset()
        print(f"[GPU {args.gpu_id}] 加载了 {len(samples)} 个样本")

        results = evaluator.evaluate(samples, model_generate_fn=model_generate_fn, batch_size=512)

        # 保存结果
        model_name = os.path.basename(MODEL_PATH)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{dataset_name}_{model_name}_{timestamp}.json"
        output_path = os.path.join(RESULTS_DIR, output_filename)

        results['model'] = model_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[GPU {args.gpu_id}] {dataset_name}: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        print(f"[GPU {args.gpu_id}] 结果已保存到: {output_path}\n")

    print(f"\n[GPU {args.gpu_id}] 所有数据集评测完成！")

if __name__ == "__main__":
    main()
