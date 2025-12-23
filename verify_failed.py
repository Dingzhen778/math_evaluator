#!/usr/bin/env python3
"""
使用 DeepSeek-R1 对所有 failed 的答案进行二次验证
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
DEEPSEEK_MODEL = "deepseek-r1"

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
                    "model": DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1024
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer EMPTY"
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']

                # 解析结果
                content_upper = content.upper()
                if "VERDICT: EQUIVALENT" in content_upper:
                    verdict = "EQUIVALENT"
                elif "VERDICT: DIFFERENT" in content_upper:
                    verdict = "DIFFERENT"
                else:
                    verdict = "AMBIGUOUS"

                return {
                    "verdict": verdict,
                    "response": content,
                    "success": True
                }
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"verdict": "ERROR", "response": str(e), "success": False}

    return {"verdict": "ERROR", "response": "Max retries exceeded", "success": False}


def load_all_results(dirs: list) -> dict:
    """加载所有结果文件"""
    all_results = {}

    for dir_path in dirs:
        for json_file in Path(dir_path).rglob("*.json"):
            if "summary" in json_file.name:
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                all_results[str(json_file)] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    return all_results


def extract_failed_samples(all_results: dict) -> list:
    """提取所有 failed 的样本"""
    failed_samples = []

    for file_path, data in all_results.items():
        if 'results' not in data:
            continue

        for idx, result in enumerate(data['results']):
            if not result.get('is_correct', True):
                failed_samples.append({
                    'file_path': file_path,
                    'index': idx,
                    'question': result.get('question', ''),
                    'reference_answer': result.get('reference_answer', ''),
                    'generated_solution': result.get('generated_solution', ''),
                    'predicted_answer': result.get('predicted_answer', ''),
                    'model': data.get('model', ''),
                    'dataset': data.get('dataset', '')
                })

    return failed_samples


def verify_sample(sample: dict) -> dict:
    """验证单个样本"""
    result = call_deepseek_r1(
        problem=sample['question'],
        answer=sample['reference_answer'],
        generation=sample['generated_solution']
    )

    sample['verification'] = result
    sample['verified_correct'] = result['verdict'] == 'EQUIVALENT'

    return sample


def main():
    # 输入目录
    input_dirs = [
        "/volume/data/rhjiang/math_evaluator/results-1222",
        "/volume/data/rhjiang/math_evaluator/results-1222-new"
    ]

    # 输出目录
    output_dir = Path("/volume/data/rhjiang/math_evaluator/results-1222-verified")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("加载所有结果文件...")
    print("=" * 60)

    all_results = load_all_results(input_dirs)
    print(f"共加载 {len(all_results)} 个结果文件")

    print("\n提取 failed 样本...")
    failed_samples = extract_failed_samples(all_results)
    print(f"共 {len(failed_samples)} 个 failed 样本需要验证")

    if len(failed_samples) == 0:
        print("没有 failed 样本需要验证")
        return

    print("\n" + "=" * 60)
    print("开始 DeepSeek-R1 二次验证...")
    print("=" * 60)

    # 并行验证
    verified_samples = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(verify_sample, s): s for s in failed_samples}

        for future in tqdm(as_completed(futures), total=len(failed_samples), desc="验证进度"):
            try:
                result = future.result()
                verified_samples.append(result)
            except Exception as e:
                print(f"Error: {e}")

    # 保存验证结果
    verification_results = defaultdict(lambda: defaultdict(list))

    for sample in verified_samples:
        key = f"{sample['dataset']}_{sample['model']}"
        verification_results[sample['model']][sample['dataset']].append({
            'question': sample['question'][:100] + '...',
            'reference_answer': sample['reference_answer'],
            'predicted_answer': sample['predicted_answer'],
            'verdict': sample['verification']['verdict'],
            'verified_correct': sample['verified_correct']
        })

    # 保存详细验证结果
    with open(output_dir / "verification_details.json", 'w') as f:
        json.dump(verified_samples, f, indent=2, ensure_ascii=False)

    print(f"\n验证结果已保存到: {output_dir / 'verification_details.json'}")

    # 统计结果
    print("\n" + "=" * 60)
    print("统计最终结果...")
    print("=" * 60)

    # 按模型和数据集统计
    stats = defaultdict(lambda: defaultdict(lambda: {
        'total': 0,
        'original_correct': 0,
        'verified_correct': 0,
        'rounds': defaultdict(lambda: {'total': 0, 'original_correct': 0, 'verified_correct': 0})
    }))

    # 统计原始正确的
    for file_path, data in all_results.items():
        model = data.get('model', '')
        dataset = data.get('dataset', '')

        # 从文件路径提取 round
        round_num = 1
        if 'round' in file_path:
            try:
                round_num = int(file_path.split('round')[1].split('/')[0])
            except:
                pass

        if 'results' not in data:
            continue

        for result in data['results']:
            stats[model][dataset]['total'] += 1
            stats[model][dataset]['rounds'][round_num]['total'] += 1

            if result.get('is_correct', False):
                stats[model][dataset]['original_correct'] += 1
                stats[model][dataset]['rounds'][round_num]['original_correct'] += 1

    # 统计验证后正确的
    for sample in verified_samples:
        model = sample['model']
        dataset = sample['dataset']

        # 从文件路径提取 round
        round_num = 1
        if 'round' in sample['file_path']:
            try:
                round_num = int(sample['file_path'].split('round')[1].split('/')[0])
            except:
                pass

        if sample['verified_correct']:
            stats[model][dataset]['verified_correct'] += 1
            stats[model][dataset]['rounds'][round_num]['verified_correct'] += 1

    # 计算最终准确率
    final_results = defaultdict(dict)

    for model in sorted(stats.keys()):
        for dataset in sorted(stats[model].keys()):
            s = stats[model][dataset]

            # 计算5轮平均
            round_accuracies = []
            for round_num in range(1, 6):
                r = s['rounds'][round_num]
                if r['total'] > 0:
                    original_acc = r['original_correct'] / r['total'] * 100
                    verified_acc = (r['original_correct'] + r['verified_correct']) / r['total'] * 100
                    round_accuracies.append({
                        'round': round_num,
                        'original': original_acc,
                        'final': verified_acc
                    })

            if round_accuracies:
                avg_original = sum(r['original'] for r in round_accuracies) / len(round_accuracies)
                avg_final = sum(r['final'] for r in round_accuracies) / len(round_accuracies)

                final_results[model][dataset] = {
                    'original_acc': round(avg_original, 2),
                    'final_acc': round(avg_final, 2),
                    'improvement': round(avg_final - avg_original, 2)
                }

    # 保存最终统计
    with open(output_dir / "final_summary.json", 'w') as f:
        json.dump(dict(final_results), f, indent=2, ensure_ascii=False)

    # 打印结果表格
    print("\n" + "=" * 80)
    print("最终结果 (5轮平均)")
    print("=" * 80)

    # 获取所有数据集
    all_datasets = set()
    for model_data in final_results.values():
        all_datasets.update(model_data.keys())
    all_datasets = sorted(all_datasets)

    # 打印表头
    header = "模型".ljust(45) + " | " + " | ".join(ds[:12].ljust(12) for ds in all_datasets)
    print(header)
    print("-" * len(header))

    for model in sorted(final_results.keys()):
        row = model[:44].ljust(45) + " | "
        for ds in all_datasets:
            if ds in final_results[model]:
                orig = final_results[model][ds]['original_acc']
                final = final_results[model][ds]['final_acc']
                cell = f"{orig:.1f}→{final:.1f}"
            else:
                cell = "-"
            row += cell.ljust(12) + " | "
        print(row)

    print("\n" + "=" * 80)
    print(f"结果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
