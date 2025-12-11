"""
MATH Answer Validator
参考自opencompass和eval_base_model的实现
"""

import re
from typing import Optional
from ..base import BaseAnswerValidator


class MATHAnswerValidator(BaseAnswerValidator):
    """MATH数据集答案验证器"""

    def validate(self, predicted: str, reference: str) -> bool:
        """
        验证MATH答案是否正确

        策略：
        1. 规范化答案
        2. 字符串比较
        3. 数学等价性判断（简化版）

        Args:
            predicted: 预测答案
            reference: 参考答案

        Returns:
            bool: 是否正确
        """
        # 规范化答案
        pred_normalized = self._normalize_answer(predicted)
        ref_normalized = self._normalize_answer(reference)

        # 字符串相等
        if pred_normalized == ref_normalized:
            return True

        # 数值相等（如果都是数字）
        if self._is_numeric_equal(pred_normalized, ref_normalized):
            return True

        # 分数等价（简化版）
        if self._is_fraction_equal(pred_normalized, ref_normalized):
            return True

        return False

    def _normalize_answer(self, text: str) -> str:
        """
        规范化答案

        处理：
        - 移除空白字符
        - 移除LaTeX命令的多余空格
        - 统一格式
        - 移除单位
        """
        if not text:
            return ""

        # 移除首尾空白
        text = text.strip()

        # 移除LaTeX的$符号
        text = text.replace('$', '')

        # 移除常见单位词
        units = ['dollars', 'dollar', 'miles', 'mile', 'square', 'cubic',
                 'feet', 'foot', 'inches', 'inch', 'meters', 'meter',
                 'cm', 'mm', 'km', 'units', 'unit']
        for unit in units:
            # 移除单位词（区分大小写）
            text = re.sub(rf'\b{unit}\b', '', text, flags=re.IGNORECASE)

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 处理LaTeX命令
        # \\frac{a}{b} 规范化
        text = re.sub(r'\\frac\s*\{', r'\\frac{', text)

        # 移除text命令
        text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)

        # 移除特殊符号
        text = text.replace('\\,', '').replace('\\!', '')

        # 移除逗号（在数字中）
        text = re.sub(r'(\d),(\d)', r'\1\2', text)

        return text.strip()

    def _is_numeric_equal(self, str1: str, str2: str, tolerance: float = 1e-6) -> bool:
        """判断数值是否相等"""
        try:
            # 尝试转换为浮点数
            num1 = self._to_number(str1)
            num2 = self._to_number(str2)

            if num1 is None or num2 is None:
                return False

            return abs(num1 - num2) < tolerance
        except:
            return False

    def _to_number(self, text: str) -> Optional[float]:
        """将文本转换为数字"""
        # 移除空格和逗号
        text = text.replace(' ', '').replace(',', '')

        # 尝试直接转换
        try:
            return float(text)
        except:
            pass

        # 提取纯数字
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group(0))
            except:
                pass

        return None

    def _is_fraction_equal(self, str1: str, str2: str) -> bool:
        """判断分数是否相等（简化版）"""
        # 提取分数 a/b 或 \\frac{a}{b}
        frac1 = self._parse_fraction(str1)
        frac2 = self._parse_fraction(str2)

        if frac1 is None or frac2 is None:
            return False

        # 交叉相乘判断相等
        a1, b1 = frac1
        a2, b2 = frac2

        return abs(a1 * b2 - a2 * b1) < 1e-6

    def _parse_fraction(self, text: str) -> Optional[tuple]:
        """解析分数"""
        # 匹配 a/b
        match = re.search(r'(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)', text)
        if match:
            try:
                a = float(match.group(1))
                b = float(match.group(2))
                if b != 0:
                    return (a, b)
            except:
                pass

        # 匹配 \\frac{a}{b}
        match = re.search(r'\\frac\{(-?\d+\.?\d*)\}\{(-?\d+\.?\d*)\}', text)
        if match:
            try:
                a = float(match.group(1))
                b = float(match.group(2))
                if b != 0:
                    return (a, b)
            except:
                pass

        return None
