#!/usr/bin/env python3
"""
使用 DeepSeek-R1 对 Round1 的 failed 答案进行二次验证
"""

import json
import os
import requests
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# DeepSeek-R1 API 配置
DEEPSEEK_API_URL = "https://scitix-cetus.scitix.ai/siflow/cetus/hisys/rhjiang/deepseek-r1/chat/completions"

VERIFICATION_PROMPT = """You are a mathematical answer validator. You will be provided with a mathematical problem and you
need to compare the answer in the reference solution, and the final answer in a model's solution to
determine if they are equivalent, even if formatted differently.
PROBLEM:
{problem}
REFERENCE SOLUTION:
{answer}
MODEL'S SOLUTION:
{generation}
Focus ONLY on comparing the final mathematical answer provided by the model while ignoring differences in:
- Formatting (e.g., \\boxed{{}} vs plain text)
- Multiple choice formatting (e.g., "A" vs full solution)
- Order of coordinate pairs or solutions
- Equivalent mathematical expressions or notation variations
- If the model's answer is nonsense, return "Verdict: AMBIGUOUS"
Start with a brief explanation of your comparison (2-3 sentences). Then output your final answer in one of the following formats:
- "Verdict: EQUIVALENT"
- "Verdict: DIFFERENT"
- "Verdict: AMBIGUOUS"
"""


def call_deepseek_r1(problem: str, answer: str, generation: str, max_retries: int = 3) -> dict:
    """调用 DeepSeek-R1 进行验证"""
    prompt = VERIFICATION_PROMPT.format(
        problem=problem,
        answer=answer,
        generation=generation
    )

    for attempt in range(max_retries):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                json={
                    "model": "deepseek-r1",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1024
                },
                headers={"Content-Type": "application/json"},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                content_upper = content.upper()

                if "VERDICT: EQUIVALENT" in content_upper:
                    verdict = "EQUIVALENT"
                elif "VERDICT: DIFFERENT" in content_upper:
                    verdict = "DIFFERENT"
                else:
                    verdict = "AMBIGUOUS"

                return {"verdict": verdict, "success": True}
            elif response.status_code == 429:
                time.sleep(1)
                continue
            else:
                time.sleep(0.5)
                continue

        except Exception as e:
            time.sleep(0.5)
            continue

    return {"verdict": "ERROR", "success": False}


def verify_sample(sample: dict) -> dict:
    """验证单个样本"""
    result = call_deepseek_r1(
        problem=sample['question'],
        answer=sample['reference_answer'],
        generation=sample['generated_solution']
    )
    sample['verdict'] = result['verdict']
    sample['verified_correct'] = result['verdict'] == 'EQUIVALENT'
    return sample


def main():
    # 只处理 round1
    input_dirs = [
        "/volume/data/rhjiang/math_evaluator/results-1222/round1",
        "/volume/data/rhjiang/math_evaluator/results-1222-new/round1"
    ]

    output_dir = Path("/volume/data/rhjiang/math_evaluator/results-1222-verified")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("加载 Round1 结果...")
    print("=" * 60)

    # 收集 failed 样本（排除证明题）
    failed_samples = []
    all_results = {}

    for dir_path in input_dirs:
        if not os.path.exists(dir_path):
            continue
        for json_file in Path(dir_path).glob("*.json"):
            if "summary" in json_file.name:
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # 跳过证明题
                if data.get('dataset') == 'olympiadbench_tp':
                    continue

                all_results[str(json_file)] = data

                for idx, result in enumerate(data.get('results', [])):
                    if not result.get('is_correct', True):
                        failed_samples.append({
                            'file_path': str(json_file),
                            'index': idx,
                            'question': result.get('question', ''),
                            'reference_answer': result.get('reference_answer', ''),
                            'generated_solution': result.get('generated_solution', ''),
                            'predicted_answer': result.get('predicted_answer', ''),
                            'model': data.get('model', ''),
                            'dataset': data.get('dataset', '')
                        })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    print(f"共 {len(failed_samples)} 个 failed 样本需要验证")

    if len(failed_samples) == 0:
        print("没有需要验证的样本")
        return

    print(f"\n开始验证 (128 并发)...")

    # 128 并发验证
    verified_samples = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(verify_sample, s): s for s in failed_samples}

        for future in tqdm(as_completed(futures), total=len(failed_samples)):
            try:
                result = future.result()
                verified_samples.append(result)
            except Exception as e:
                print(f"Error: {e}")

    # 统计结果
    stats = defaultdict(lambda: defaultdict(lambda: {
        'total': 0, 'original_correct': 0, 'verified_extra': 0
    }))

    # 统计原始正确数
    for file_path, data in all_results.items():
        model = data.get('model', '')
        dataset = data.get('dataset', '')
        for r in data.get('results', []):
            stats[model][dataset]['total'] += 1
            if r.get('is_correct', False):
                stats[model][dataset]['original_correct'] += 1

    # 统计验证后额外正确数
    for s in verified_samples:
        if s['verified_correct']:
            stats[s['model']][s['dataset']]['verified_extra'] += 1

    # 保存结果
    with open(output_dir / "round1_verification.json", 'w') as f:
        json.dump(verified_samples, f, ensure_ascii=False)

    # 打印结果
    print("\n" + "=" * 80)
    print("Round1 验证结果")
    print("=" * 80)

    datasets = sorted(set(d for m in stats.values() for d in m.keys()))

    print(f"\n{'模型':<50} | " + " | ".join(f"{d[:10]:<10}" for d in datasets))
    print("-" * 120)

    for model in sorted(stats.keys()):
        row = f"{model[:49]:<50} | "
        for ds in datasets:
            s = stats[model][ds]
            if s['total'] > 0:
                orig = s['original_correct'] / s['total'] * 100
                final = (s['original_correct'] + s['verified_extra']) / s['total'] * 100
                row += f"{orig:.1f}→{final:.1f}".ljust(10) + " | "
            else:
                row += "-".ljust(10) + " | "
        print(row)

    # 保存统计
    summary = {}
    for model in stats:
        summary[model] = {}
        for ds in stats[model]:
            s = stats[model][ds]
            if s['total'] > 0:
                summary[model][ds] = {
                    'original': round(s['original_correct'] / s['total'] * 100, 2),
                    'final': round((s['original_correct'] + s['verified_extra']) / s['total'] * 100, 2)
                }

    with open(output_dir / "round1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
