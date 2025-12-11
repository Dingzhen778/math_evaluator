"""
AIME Answer Extractor
从模型生成的答案中提取1-999的整数答案
"""

import re
from typing import Optional
from ..base import BaseAnswerExtractor


class AImeAnswerExtractor(BaseAnswerExtractor):
    """AIME数据集答案提取器（2024和2025通用）"""

    def extract(self, text: str) -> str:
        """
        从模型输出中提取答案

        AIME答案特点：1-999的整数

        优先级：
        1. \\boxed{...}中的数字
        2. 特定模式（如"Answer: 123"）
        3. 最后一个1-999范围内的数字

        Args:
            text: 模型生成的完整输出

        Returns:
            str: 提取的答案
        """
        # 方法1: 提取\\boxed{}
        boxed_answer = self._extract_boxed(text)
        if boxed_answer:
            return boxed_answer

        # 方法2: 查找特定模式
        pattern_answer = self._extract_answer_pattern(text)
        if pattern_answer:
            return pattern_answer

        # 方法3: 提取最后一个1-999的数字
        last_valid = self._extract_last_valid_number(text)
        if last_valid:
            return last_valid

        # 兜底：提取最后一个数字
        last_number = self._extract_last_number(text)
        if last_number:
            return last_number

        return text.strip()

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\\boxed{...}中的数字"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = list(re.finditer(pattern, text))

        if not matches:
            return None

        last_match = matches[-1]
        content = last_match.group(1).strip()

        # 提取数字
        match = re.search(r'\d+', content)
        if match:
            return match.group(0)

        return content

    def _extract_answer_pattern(self, text: str) -> Optional[str]:
        """提取特定模式的答案"""
        patterns = [
            r'[Aa]nswer:?\s*(\d+)',
            r'[Tt]he answer is:?\s*(\d+)',
            r'[Ff]inal [Aa]nswer:?\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    def _extract_last_valid_number(self, text: str) -> Optional[str]:
        """提取最后一个1-999范围内的数字"""
        matches = list(re.finditer(r'\b(\d{1,3})\b', text))

        # 从后往前找第一个1-999的数字
        for match in reversed(matches):
            num = int(match.group(1))
            if 1 <= num <= 999:
                return str(num)

        return None

    def _extract_last_number(self, text: str) -> Optional[str]:
        """提取最后一个数字（无范围限制）"""
        matches = list(re.finditer(r'\d+', text))
        if matches:
            return matches[-1].group(0)

        return None
