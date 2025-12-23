#!/usr/bin/env python3
"""
使用新的 math-verify 提取逻辑重新验证 Round1 结果
"""

import json
import os
from tqdm import tqdm
from evaluators.validators.math_verify_validator import MathVerifyValidator

# 结果文件夹
RESULT_DIRS = [
    '/volume/data/rhjiang/math_evaluator/results-1222/round1',
    '/volume/data/rhjiang/math_evaluator/results-1222-new/round1',
]

OUTPUT_DIR = '/volume/data/rhjiang/math_evaluator/results-1222-mathverify/round1'

def re_evaluate_file(input_path: str, output_path: str, validator: MathVerifyValidator) -> dict:
    """重新验证单个结果文件"""
    with open(input_path) as f:
        data = json.load(f)

    results = data.get('results', [])
    correct_count = 0

    new_results = []
    for r in results:
        reference = r.get('reference_answer', '')
        solution = r.get('generated_solution', '')

        # 设置上下文并验证
        validator.set_context(problem=r.get('question', ''), generation=solution)
        is_correct = validator.validate(r.get('predicted_answer', ''), reference)

        # 获取新提取的答案
        extracted = validator.get_extracted_answer()

        if is_correct:
            correct_count += 1

        new_results.append({
            'question': r.get('question', ''),
            'reference_answer': reference,
            'predicted_answer': extracted or r.get('predicted_answer', ''),
            'generated_solution': solution,
            'is_correct': is_correct,
            'metadata': r.get('metadata', {})
        })

    # 计算新准确率
    accuracy = correct_count / len(results) * 100 if results else 0

    new_data = {
        'dataset': data.get('dataset', ''),
        'accuracy': round(accuracy, 2),
        'correct': correct_count,
        'total': len(results),
        'results': new_results
    }

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return {
        'file': os.path.basename(input_path),
        'old_accuracy': data.get('accuracy', 0),
        'new_accuracy': round(accuracy, 2),
        'old_correct': data.get('correct', 0),
        'new_correct': correct_count,
        'total': len(results)
    }

def main():
    validator = MathVerifyValidator(use_math_verify_extraction=True)

    all_results = []

    for result_dir in RESULT_DIRS:
        if not os.path.exists(result_dir):
            continue

        files = [f for f in os.listdir(result_dir) if f.endswith('.json')]

        for fname in tqdm(files, desc=f"Processing {result_dir.split('/')[-2]}"):
            input_path = os.path.join(result_dir, fname)
            output_path = os.path.join(OUTPUT_DIR, fname)

            try:
                result = re_evaluate_file(input_path, output_path, validator)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

    # 打印对比结果
    print("\n" + "=" * 80)
    print("Round1 重新验证结果对比")
    print("=" * 80)

    # 按数据集分组
    by_dataset = {}
    for r in all_results:
        ds = r['file'].split('_Qwen')[0]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)

    total_old_correct = 0
    total_new_correct = 0
    total_samples = 0

    for ds, results in sorted(by_dataset.items()):
        print(f"\n### {ds}")
        print("-" * 60)

        for r in results:
            model = r['file'].split('_Qwen2.5-math-1.5B-numinamath-')[1].split('_')[0]
            diff = r['new_correct'] - r['old_correct']
            diff_str = f"+{diff}" if diff > 0 else str(diff)

            print(f"  {model}: {r['old_accuracy']:.2f}% -> {r['new_accuracy']:.2f}% ({diff_str})")

            total_old_correct += r['old_correct']
            total_new_correct += r['new_correct']
            total_samples += r['total']

    print("\n" + "=" * 80)
    print(f"总计: {total_old_correct} -> {total_new_correct} (+{total_new_correct - total_old_correct})")
    print(f"总准确率: {total_old_correct/total_samples*100:.2f}% -> {total_new_correct/total_samples*100:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()
