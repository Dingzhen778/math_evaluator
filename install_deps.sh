#!/bin/bash
# 安装依赖脚本
# 由于容器环境每次重启可能重置，需要重新安装依赖
# 直接使用系统 Python（已包含 torch 2.8.0a0）

set -e

echo "=========================================="
echo "安装 math_evaluator 依赖"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

echo "Python 版本: $(python3 --version)"
echo "Python 路径: $(which python3)"
echo ""

# 安装基础依赖（除了 vllm）
echo "安装基础依赖..."
python3 -m pip install --quiet \
    "tqdm>=4.65.0" \
    "requests>=2.31.0" \
    "math-verify>=0.8.0" \
    "datasets>=2.14.0" \
    "transformers>=4.57.0"

echo "✓ 基础依赖安装完成"
echo ""

# 安装 vllm（使用 --no-deps 避免 torch 版本冲突）
echo "安装 vllm==0.11.0 (使用 --no-deps 避免 torch 版本冲突)..."
python3 -m pip install --quiet vllm==0.11.0 --no-deps

echo "✓ vllm 安装完成"
echo ""

# 验证安装
echo "=========================================="
echo "验证安装"
echo "=========================================="
python3 -c "
import sys
packages = ['tqdm', 'requests', 'math_verify', 'datasets', 'transformers', 'vllm', 'torch']
missing = []
for pkg in packages:
    try:
        if pkg == 'math_verify':
            __import__('math_verify')
        else:
            __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} (未安装)')
        missing.append(pkg)

if missing:
    print(f'\n错误: 以下包未安装: {missing}')
    sys.exit(1)
else:
    print('\n✓ 所有依赖已正确安装')
"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "现在可以直接运行脚本，例如:"
echo "  python3 run_eval.py ..."
echo "  bash batch_eval_rounds.sh"
echo ""







