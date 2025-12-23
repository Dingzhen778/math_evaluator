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
from evaluators.vllm_client import VLLMClient




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
    base_url: str = None,
    model: str = None,
    model_path: str = None,
    api_key: str = "EMPTY",
    max_samples: int = None,
    max_workers: int = 32,
    use_few_shot: bool = True,
    gpu_ids: str = "0",
    vllm_client=None,  # 可选的已加载的 vLLM 客户端
    model_generate_fn=None,  # 可选的已创建的生成函数
    results_dir: str = "results",  # 结果保存目录
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
    
    # 优先使用传入的已加载模型，否则加载新模型
    if vllm_client is not None and model_generate_fn is not None:
        # 使用已加载的模型（复用）
        print(f"数据集: {dataset_name}")
        print(f"使用方式: vLLM Python API (复用已加载的模型)")
        print(f"GPU: {gpu_ids}")
    elif model_path:
        # 需要加载新模型（单数据集模式）
        print(f"模型路径: {model_path}")
        print(f"使用方式: vLLM Python API (直接，批量处理)")
        print(f"GPU: {gpu_ids}")
        
        # 直接使用 vLLM Python API（支持多GPU tensor parallelism）
        # 解析 GPU IDs 数量
        num_gpus = len([gpu.strip() for gpu in gpu_ids.split(",")])
        
        vllm_client = VLLMClient(
            model_path=model_path,
            gpu_ids=gpu_ids,
            gpu_memory_utilization=0.95,  # 拉满利用率
            max_model_len=4096,  # 适配模型 max_position_embeddings
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,  # 使用所有GPU进行tensor parallelism
        )
        
        # 使用批量处理函数（更高效）
        def model_generate_fn(system_prompt: str, user_prompt: str) -> str:
            return vllm_client.generate(system_prompt, user_prompt)
    else:
        # 使用 HTTP API
        print(f"模型: {model}")
        print(f"API: {base_url}")
        api_client = APIClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_retries=1,  # 快速失败，只重试1次
            timeout=120
        )
        
        def model_generate_fn(system_prompt: str, user_prompt: str) -> str:
            return api_client.generate(system_prompt, user_prompt)
    
    print(f"Few-shot: {use_few_shot}")
    print(f"最大样本数: {max_samples or '全部'}")
    print(f"并行线程数: {max_workers}")
    print("=" * 80)

    evaluator = MathEvaluator(
        dataset_name=dataset_name,
        model_generate_fn=model_generate_fn,
        max_workers=max_workers,
        use_few_shot=use_few_shot,
        vllm_client=vllm_client  # 传递 vllm_client 对象以启用批量处理
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_path:
        model_name = os.path.basename(model_path)
    else:
        model_name = model.replace('/', '_')
    output_filename = f"{dataset_name}_{model_name}_{timestamp}.json"
    
    # 使用传入的结果目录
    output_path = os.path.join(results_dir, output_filename)

    os.makedirs(results_dir, exist_ok=True)

    print(f"\n开始评测...")
    # 获取模型名称
    if model_path:
        model_name_for_result = os.path.basename(model_path)
    else:
        model_name_for_result = model.replace('/', '_')
    
    results = evaluator.evaluate_and_save(
        output_path=output_path,
        max_samples=max_samples,
        model_name=model_name_for_result
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
        choices=['math500', 'gsm8k', 'aime2024', 'aime2025', 'gaokao2023en', 'mathodyssey', 'amc23', 'olympiadbench_oe', 'olympiadbench_tp', 'both', 'all'],
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
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='模型路径（直接使用 vLLM Python API）'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU ID（例如 "0" 或 "0,1"）'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='结果保存目录（默认: results）'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=None,
        help='Tensor 并行大小（默认: 自动根据GPU数量设置）'
    )

    args = parser.parse_args()

    config = get_config()

    if args.base_url:
        config['base_url'] = args.base_url
    if args.model:
        config['model'] = args.model
    
    # 确定使用哪种方式
    model_path = args.model_path
    base_url = config['base_url'] if not model_path else None
    model = config['model'] if not model_path else None
    
    # 获取结果目录
    results_dir = args.results_dir
    
    # 如果使用 vLLM 直接模式，先加载模型（只加载一次，用于所有数据集）
    vllm_client_obj = None
    model_generate_fn = None
    
    if model_path:
        print("=" * 80)
        print("正在加载模型（将用于所有数据集）...")
        print("=" * 80)

        # 解析 GPU IDs 数量
        num_gpus = len([gpu.strip() for gpu in args.gpu.split(",")])
        # 使用指定的 tensor_parallel_size，否则使用 GPU 数量
        tp_size = args.tensor_parallel_size if args.tensor_parallel_size else num_gpus

        vllm_client_obj = VLLMClient(
            model_path=model_path,
            gpu_ids=args.gpu,
            gpu_memory_utilization=0.95,  # 拉满利用率
            max_model_len=4096,  # 适配模型 max_position_embeddings
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
        )
        
        # 使用批量处理函数（更高效）
        def model_generate_fn(system_prompt: str, user_prompt: str) -> str:
            return vllm_client_obj.generate(system_prompt, user_prompt)
        
        print("=" * 80)
        print("模型加载完成！")
        print("=" * 80)
        print("")
        
    if args.dataset == 'all':
        # 评测所有数据集，复用已加载的模型
        for dataset_name in ['math500', 'gsm8k', 'aime2024', 'aime2025']:
            run_evaluation(
                dataset_name=dataset_name,
                base_url=base_url,
                model=model,
                model_path=model_path,
                api_key=config['api_key'],
                max_samples=args.max_samples,
                max_workers=args.max_workers,
                use_few_shot=not args.no_few_shot,
                gpu_ids=args.gpu,
                vllm_client=vllm_client_obj,  # 传递已加载的模型
                model_generate_fn=model_generate_fn,  # 传递生成函数
                results_dir=results_dir  # 传递结果目录
            )
            print("\n")
    else:
        run_evaluation(
            dataset_name=args.dataset,
            base_url=base_url,
            model=model,
            model_path=model_path,
            api_key=config['api_key'],
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            use_few_shot=not args.no_few_shot,
            gpu_ids=args.gpu,
            vllm_client=vllm_client_obj,  # 传递已加载的模型
            model_generate_fn=model_generate_fn,  # 传递生成函数
            results_dir=results_dir  # 传递结果目录
        )


if __name__ == "__main__":
    main()
