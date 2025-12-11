"""
MATH Dataset Loader
加载位置: evaluators/datasets/math/math.json
"""

import json
import re
import random
import os
from typing import List, Optional
from ..base import BaseDatasetLoader, EvalSample


class MATHDatasetLoader(BaseDatasetLoader):
    """MATH数据集加载器"""

    def __init__(self):
        # 使用相对于当前文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_path = os.path.join(current_dir, 'math', 'math.json')

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None,
             seed: int = 42) -> List[EvalSample]:
        """
        加载MATH数据集

        Args:
            path: 数据集路径，默认使用预设路径
            max_samples: 最大样本数（用于采样MATH500）
            seed: 随机种子

        Returns:
            List[EvalSample]: 评测样本列表
        """
        if path is None:
            path = self.default_path

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for key, item in data.items():
            # 从solution中提取boxed答案
            answer = self._extract_boxed_answer(item['solution'])

            sample = EvalSample(
                question=item['problem'],
                answer=answer,
                metadata={
                    'level': item['level'],
                    'type': item['type'],
                    'key': key,
                    'full_solution': item['solution']
                }
            )
            samples.append(sample)

        # 如果指定max_samples，进行采样（MATH500）
        if max_samples is not None and max_samples < len(samples):
            random.seed(seed)
            samples = random.sample(samples, max_samples)

        return samples

    def _extract_boxed_answer(self, text: str) -> str:
        """从solution中提取\\boxed{}中的答案"""
        # 查找所有\\boxed{...}
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = list(re.finditer(pattern, text))

        if not matches:
            return ""

        # 取最后一个boxed
        last_match = matches[-1]
        answer = last_match.group(1)

        # 处理双重花括号：\\boxed{{2}} -> 2
        if answer.startswith('{') and answer.endswith('}'):
            answer = answer[1:-1]

        return answer.strip()
