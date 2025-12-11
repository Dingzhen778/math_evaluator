"""
AIME Answer Validator using math-verify library for symbolic comparison
"""

from typing import Optional
from ..base import BaseAnswerValidator

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math-verify not installed. Install with: pip install math-verify")


class AIMEAnswerValidator(BaseAnswerValidator):
    """
    AIME答案验证器，使用math-verify库进行符号数学验证

    math-verify库支持：
    - 符号表达式比较
    - LaTeX公式解析
    - 数值和代数等价性检查
    """

    def __init__(self):
        if not MATH_VERIFY_AVAILABLE:
            raise ImportError(
                "math-verify library is required for AIME validation. "
                "Install it with: pip install math-verify"
            )

    def validate(self, predicted: str, reference: str) -> bool:
        """
        验证预测答案和参考答案是否相等（实现抽象方法）

        Args:
            predicted: 模型预测的答案
            reference: 参考答案

        Returns:
            bool: 是否相等
        """
        return self.is_equal(predicted, reference)

    def is_equal(self, predicted: str, reference: str) -> bool:
        """
        使用math-verify验证预测答案和参考答案是否相等

        Args:
            predicted: 模型预测的答案
            reference: 参考答案

        Returns:
            bool: 是否相等
        """
        if not predicted or not reference:
            return False

        try:
            # 确保答案被LaTeX环境包围（math-verify要求）
            prediction_with_env = f"${predicted}$"
            reference_with_env = f"${reference}$"

            # 使用math-verify解析和验证
            parsed_prediction = parse(prediction_with_env)
            parsed_reference = parse(reference_with_env)

            is_correct = verify(parsed_prediction, parsed_reference)

            return is_correct

        except Exception as e:
            # 如果math-verify解析失败，尝试简单的字符串/数值比较作为fallback
            return self._fallback_comparison(predicted, reference)

    def _fallback_comparison(self, predicted: str, reference: str) -> bool:
        """
        当math-verify失败时的fallback比较方法

        Args:
            predicted: 预测答案
            reference: 参考答案

        Returns:
            bool: 是否相等
        """
        # 清理字符串
        pred_clean = predicted.strip().lower()
        ref_clean = reference.strip().lower()

        # 字符串完全匹配
        if pred_clean == ref_clean:
            return True

        # 尝试数值比较
        try:
            pred_num = float(pred_clean)
            ref_num = float(ref_clean)
            # AIME通常是整数答案，使用严格相等
            return abs(pred_num - ref_num) < 1e-9
        except (ValueError, TypeError):
            pass

        return False
