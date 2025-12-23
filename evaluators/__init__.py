"""
Math Evaluation Framework for Instruct Models
支持 MATH500, GSM8K, AIME24, AIME25 数据集评测
"""

from .evaluator import MathEvaluator

# vLLM 是可选依赖
try:
    from .vllm_client import VLLMClient
    __all__ = ['MathEvaluator', 'VLLMClient']
except ImportError:
    __all__ = ['MathEvaluator']
