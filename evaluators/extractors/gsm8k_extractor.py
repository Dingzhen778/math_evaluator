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
        从模型输出中提取数值答案（原始简单版本）
        
        策略：
        1. 如果有####标记，提取最后一个####后面的第一个数字
        2. 否则，提取文本中的最后一个数字

        Args:
            text: 模型生成的完整输出

        Returns:
            str: 提取的数值答案
        """
        if not text or not text.strip():
            return text.strip()
        
        # 方法1: 提取####后的数字（GSM8K标准格式）
        if '####' in text:
            # 使用split('####')[-1]取最后一个####后面的内容（原始逻辑）
            answer_part = text.split('####')[-1].strip()
            # 提取第一个数字（包括负数和小数）
            match = re.search(r'-?\d+\.?\d*', answer_part)
            if match:
                return match.group(0).replace(',', '')
        
        # 方法2: 提取最后一个数字（回退方案）
        # 先移除"Question:"之后的内容，避免提取到新问题
        text_before_question = text.split('Question:')[0]
        numbers = re.findall(r'-?\d+\.?\d+|-?\d+', text_before_question)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return text.strip()
