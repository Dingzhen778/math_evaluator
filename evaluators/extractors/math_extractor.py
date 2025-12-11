"""
MATH Answer Extractor
从模型生成的答案中提取最终答案
"""

import re
from typing import Optional
from ..base import BaseAnswerExtractor


class MATHAnswerExtractor(BaseAnswerExtractor):
    """MATH数据集答案提取器"""

    def extract(self, text: str) -> str:
        """
        从模型输出中提取答案

        优先级：
        1. \\boxed{...}
        2. "The final answer is ..." 或 "Final Answer: ..."
        3. 最后一行非空文本

        Args:
            text: 模型生成的完整输出

        Returns:
            str: 提取的答案
        """
        # 方法1: 提取\\boxed{}
        boxed_answer = self._extract_boxed(text)
        if boxed_answer:
            return boxed_answer

        # 方法2: 查找"The final answer is"模式
        final_answer = self._extract_final_answer_pattern(text)
        if final_answer:
            return final_answer

        # 方法3: 返回最后一行非空文本
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            return lines[-1]

        return text.strip()

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\\boxed{...}中的内容"""
        # 正则匹配\\boxed{...}，支持嵌套花括号
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = list(re.finditer(pattern, text))

        if not matches:
            return None

        # 取最后一个boxed
        last_match = matches[-1]
        answer = last_match.group(1)

        # 处理双重花括号：\\boxed{{2}} -> 2
        if answer.startswith('{') and answer.endswith('}'):
            answer = answer[1:-1]

        return answer.strip()

    def _extract_final_answer_pattern(self, text: str) -> Optional[str]:
        """提取"The final answer is ..."模式"""
        patterns = [
            r'[Tt]he final answer is:?\s*(.+)',
            r'[Ff]inal [Aa]nswer:?\s*(.+)',
            r'[Tt]herefore,?\s+(?:the answer is:?\s*)?(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # 移除末尾的句号
                answer = answer.rstrip('.')
                return answer

        return None
