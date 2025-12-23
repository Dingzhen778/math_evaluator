#!/usr/bin/env python3
"""
批量评测，自动部署vLLM并评测
"""

import os
import sys
import subprocess
import argparse
import time
import glob
import requests
from pathlib import Path


def check_vllm_installed():
    try:
        subprocess.run(["vllm", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def start_vllm_server(model_path: str, port: int = 8000, gpu_ids: str = "0"):
    model_name = os.path.basename(model_path)

    print(f"\n{'='*80}")
    print(f"启动vLLM服务器")
    print(f"{'='*80}")
    print(f"模型路径: {model_path}")
    print(f"模型名称: {model_name}")
    print(f"端口: {port}")
    print(f"GPU: {gpu_ids}")
    print(f"{'='*80}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    cmd = [
        "vllm", "serve",
        model_path,
        "--port", str(port),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "8192"
    ]

    print(f"执行命令: {' '.join(cmd)}\n")

    #启动vLLM
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process


def wait_for_server(base_url: str, max_retries: int = 60, retry_interval: int = 5):

    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                print(f"✓ vLLM已启动\n")
                return True
        except requests.exceptions.RequestException:
            pass

        print(f"  尝试 {i+1}/{max_retries}...")
        time.sleep(retry_interval)

    print(f"✗ vLLM服务器启动超时")
    return False


def run_evaluation(model_path: str, base_url: str, datasets: str = "all", max_workers: int = 32, use_vllm_direct: bool = True, gpu_ids: str = "0", results_dir: str = "results"):
    """运行评测"""
    model_name = os.path.basename(model_path)

    print(f"\n{'='*80}")
    print(f"开始评测")
    print(f"{'='*80}")
    print(f"模型: {model_name}")
    print(f"数据集: {datasets}")
    print(f"并行数: {max_workers}")
    print(f"GPU: {gpu_ids}")
    print(f"结果目录: {results_dir}")
    print(f"使用方式: {'vLLM Python API (直接)' if use_vllm_direct else 'HTTP API'}")
    print(f"{'='*80}\n")

    if use_vllm_direct:
        # 直接使用 vLLM Python API
        cmd = [
            "python", "run_eval.py",
            "--dataset", datasets,
            "--model-path", model_path,
            "--gpu", gpu_ids,
            "--max-workers", str(max_workers),
            "--results-dir", results_dir
        ]
    else:
        # 使用 HTTP API
        cmd = [
            "python", "run_eval.py",
            "--dataset", datasets,
            "--base-url", base_url,
            "--model", model_name,
            "--max-workers", str(max_workers)
        ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ 评测完成: {model_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 评测失败: {model_name}")
        print(f"错误: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量评测模型")
    parser.add_argument(
        "--model-path",
        type=str,
        help="单个模型路径"
    )
    parser.add_argument(
        "--model-dir",
        type=str
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID，可以使用多个GPU，例如 '0,1,2,3' 使用4个GPU"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        choices=["math500", "gsm8k", "aime2024", "aime2025", "all"]
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="并行线程数 (默认: 32)"
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="结果保存目录（默认: results）"
    )

    args = parser.parse_args()

    # 检查vLLM安装
    if not args.skip_vllm and not check_vllm_installed():
        print("错误: 未安装vLLM")
        print("安装命令: pip install vllm")
        sys.exit(1)

    # 获取模型路径列表
    model_paths = []
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"错误: 模型路径不存在: {args.model_path}")
            sys.exit(1)
        model_paths = [args.model_path]
    elif args.model_dir:
        pattern = os.path.join(args.model_dir, args.pattern)
        model_paths = sorted(glob.glob(pattern))
        if not model_paths:
            print(f"错误: 未找到匹配的模型: {pattern}")
            sys.exit(1)
    else:
        print("错误: 必须指定 --model-path 或 --model-dir")
        parser.print_help()
        sys.exit(1)

    print(f"\n找到 {len(model_paths)} 个模型:")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print()

    # 评测每个模型（直接使用 vLLM Python API，不需要 HTTP 服务器）
    results = []

    for i, model_path in enumerate(model_paths, 1):
        print(f"\n{'#'*80}")
        print(f"# 评测进度: {i}/{len(model_paths)}")
        print(f"{'#'*80}\n")

        try:
            # 直接使用 vLLM Python API，不需要启动 HTTP 服务器
            # base_url 参数在这里不会被使用，但保留以兼容函数签名
            base_url = f"http://localhost:{args.port}/v1"  # 不会被使用
            results_dir = getattr(args, 'results_dir', 'results')
            success = run_evaluation(model_path, base_url, args.datasets, args.max_workers, use_vllm_direct=True, gpu_ids=args.gpu, results_dir=results_dir)
            results.append((model_path, success))

        except Exception as e:
            print(f"\n✗ 评测失败: {os.path.basename(model_path)}")
            print(f"错误: {e}\n")
            results.append((model_path, False))

    # 打印汇总
    print(f"\n{'='*80}")
    print(f"评测汇总")
    print(f"{'='*80}")
    for model_path, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {os.path.basename(model_path)}")
    print(f"{'='*80}\n")

    # 成功数量
    success_count = sum(1 for _, success in results if success)
    print(f"完成: {success_count}/{len(results)}")
    print(f"结果保存在: results/\n")


if __name__ == "__main__":
    main()
