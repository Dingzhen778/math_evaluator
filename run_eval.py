#!/usr/bin/env python3
"""
python run_eval.py --dataset [math500|gsm8k|aime2024|aime2025|all]
- EVAL_BASE_URL: API基础URL
- EVAL_MODEL: 模型名称
- EVAL_API_KEY: API密钥（默认EMPTY）
"""

import os
import argparse
from datetime import datetime
from evaluators import MathEvaluator
from evaluators.api_client import APIClient




DEFAULT_BASE_URL = "https://console.scitix.ai/siflow/cetus/hisys/ylsun/eval-qwen2-5-72b-instruct/v1"
DEFAULT_MODEL = "eval-qwen2-5-72b-instruct"
DEFAULT_API_KEY = "EMPTY"

def get_config():
    base_url = os.environ.get('EVAL_BASE_URL', DEFAULT_BASE_URL)
    model = os.environ.get('EVAL_MODEL', DEFAULT_MODEL)
    api_key = os.environ.get('EVAL_API_KEY', DEFAULT_API_KEY)

    return {
        'base_url': base_url,
        'model': model,
        'api_key': api_key
    }


def run_evaluation(
    dataset_name: str,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    max_samples: int = None,
    max_workers: int = 32,
    use_few_shot: bool = True,
):
    """
    运行评测

    Args:
        dataset_name: 数据集名称 ('math500', 'gsm8k', 'aime2024', 'aime2025')
        base_url: API基础URL
        model: 模型名称
        api_key: API密钥
        max_samples: 最大样本数（None表示全部）
        max_workers: 并行线程数
        use_few_shot: 是否使用few-shot
    """
    print("=" * 80)
    print(f"评测配置")
    print("=" * 80)
    print(f"数据集: {dataset_name}")
    print(f"模型: {model}")
    print(f"API: {base_url}")
    print(f"Few-shot: {use_few_shot}")
    print(f"最大样本数: {max_samples or '全部'}")
    print(f"并行线程数: {max_workers}")
    print("=" * 80)

    api_client = APIClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_retries=3,
        timeout=120
    )

    def model_generate_fn(system_prompt: str, user_prompt: str) -> str:
        return api_client.generate(system_prompt, user_prompt)

    evaluator = MathEvaluator(
        dataset_name=dataset_name,
        model_generate_fn=model_generate_fn,
        max_workers=max_workers,
        use_few_shot=use_few_shot
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{dataset_name}_{model.replace('/', '_')}_{timestamp}.json"
    output_path = os.path.join('results', output_filename)

    os.makedirs('results', exist_ok=True)

    print(f"\n开始评测...")
    results = evaluator.evaluate_and_save(
        output_path=output_path,
        max_samples=max_samples
    )

    print("\n" + "=" * 80)
    print(f"评测完成！")
    print("=" * 80)
    print(f"数据集: {dataset_name}")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"正确数: {results['correct']}/{results['total']}")
    print(f"结果已保存到: {output_path}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='运行数学评测')
    parser.add_argument(
        '--dataset',
        type=str,
        default='math500',
        choices=['math500', 'gsm8k', 'aime2024', 'aime2025', 'both', 'all'],
        help='数据集'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大样本数（用于快速测试）'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=32,
        help='并行线程数 (默认: 32)'
    )
    parser.add_argument(
        '--no-few-shot',
        action='store_true',
        help='禁用few-shot示例'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='API基础URL（覆盖默认配置）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='模型名称（覆盖默认配置）'
    )

    args = parser.parse_args()

    config = get_config()

    if args.base_url:
        config['base_url'] = args.base_url
    if args.model:
        config['model'] = args.model
        
    if args.dataset == 'all':
        for dataset_name in ['math500', 'gsm8k', 'aime2024', 'aime2025']:
            run_evaluation(
                dataset_name=dataset_name,
                base_url=config['base_url'],
                model=config['model'],
                api_key=config['api_key'],
                max_samples=args.max_samples,
                max_workers=args.max_workers,
                use_few_shot=not args.no_few_shot
            )
            print("\n")
    else:
        run_evaluation(
            dataset_name=args.dataset,
            base_url=config['base_url'],
            model=config['model'],
            api_key=config['api_key'],
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_few_shot=not args.no_few_shot
        )


if __name__ == "__main__":
    main()
