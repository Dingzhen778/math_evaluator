"""
GSM8K Answer Extractor
从模型生成的答案中提取数值答案
"""

import re
from typing import Optional
from ..base import BaseAnswerExtractor


class GSM8KAnswerExtractor(BaseAnswerExtractor):
    """GSM8K数据集答案提取器"""

    def extract(self, text: str) -> str:
        """
        从模型输出中提取数值答案

        优先级：
        1. #### 后的数字
        2. \\boxed{...}
        3. 最后一个数字

        Args:
            text: 模型生成的完整输出

        Returns:
            str: 提取的数值答案
        """
        # 方法1: 提取#### 后的数字
        if '####' in text:
            answer = self._extract_after_marker(text, '####')
            if answer:
                return answer

        # 方法2: 提取\\boxed{}
        boxed_answer = self._extract_boxed(text)
        if boxed_answer:
            return boxed_answer

        # 方法3: 提取最后一个数字
        last_number = self._extract_last_number(text)
        if last_number:
            return last_number

        return text.strip()

    def _extract_after_marker(self, text: str, marker: str) -> Optional[str]:
        """提取标记后的第一个数字"""
        parts = text.split(marker)
        if len(parts) < 2:
            return None

        answer_part = parts[-1].strip()
        # 提取第一个数字（支持负数、小数、逗号分隔）
        match = re.search(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', answer_part)
        if match:
            # 移除逗号
            return match.group(0).replace(',', '')

        return None

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\\boxed{...}中的数字"""
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = list(re.finditer(pattern, text))

        if not matches:
            return None

        last_match = matches[-1]
        content = last_match.group(1)

        # 从content中提取数字
        match = re.search(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', content)
        if match:
            return match.group(0).replace(',', '')

        return content.strip()

    def _extract_last_number(self, text: str) -> Optional[str]:
        """提取文本中最后一个数字"""
        # 查找所有数字
        matches = list(re.finditer(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', text))
        if matches:
            return matches[-1].group(0).replace(',', '')

        return None
