"""
Numeric Answer Validator
用于GSM8K和AIME数据集
"""

import re
from typing import Optional
from ..base import BaseAnswerValidator


class NumericAnswerValidator(BaseAnswerValidator):
    """数值答案验证器（用于GSM8K和AIME）"""

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: 数值比较的容差
        """
        self.tolerance = tolerance

    def validate(self, predicted: str, reference: str) -> bool:
        """
        验证数值答案是否正确

        策略：
        1. 字符串完全相等
        2. 数值相等（浮点数容差）
        3. 整数相等

        Args:
            predicted: 预测答案
            reference: 参考答案

        Returns:
            bool: 是否正确
        """
        # 预处理
        pred = self._preprocess(predicted)
        ref = self._preprocess(reference)

        # 字符串相等
        if pred == ref:
            return True

        # 数值相等
        pred_num = self._to_number(pred)
        ref_num = self._to_number(ref)

        if pred_num is not None and ref_num is not None:
            # 浮点数比较
            if abs(pred_num - ref_num) < self.tolerance:
                return True

            # 整数比较（四舍五入）
            if abs(round(pred_num) - round(ref_num)) < 1e-9:
                return True

        return False

    def _preprocess(self, text: str) -> str:
        """预处理答案文本"""
        if not text:
            return ""

        # 移除空白
        text = text.strip()

        # 移除逗号（千位分隔符）
        text = text.replace(',', '')

        # 移除美元符号
        text = text.replace('$', '')

        # 移除百分号（转换为小数）
        if '%' in text:
            text = text.replace('%', '')
            try:
                num = float(text)
                text = str(num / 100)
            except:
                pass

        return text

    def _to_number(self, text: str) -> Optional[float]:
        """将文本转换为数字"""
        # 移除空格
        text = text.replace(' ', '')

        # 尝试直接转换
        try:
            return float(text)
        except:
            pass

        # 提取第一个数字
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group(0))
            except:
                pass

        return None
