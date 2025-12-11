"""
GSM8K Dataset Loader
加载位置: evaluators/datasets/gsm8k/test.jsonl
"""

import json
import random
import re
import os
from typing import List, Optional
from ..base import BaseDatasetLoader, EvalSample


class GSM8KDatasetLoader(BaseDatasetLoader):
    """GSM8K数据集加载器"""

    def __init__(self):
        # 使用相对于当前文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_path = os.path.join(current_dir, 'gsm8k', 'test.jsonl')

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None,
             seed: int = 42) -> List[EvalSample]:
        """
        加载GSM8K数据集

        Args:
            path: 数据集路径，默认使用预设路径
            max_samples: 最大样本数
            seed: 随机种子

        Returns:
            List[EvalSample]: 评测样本列表
        """
        if path is None:
            path = self.default_path

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 从answer字段提取####后的数字
                answer = self._extract_answer(item['answer'])

                sample = EvalSample(
                    question=item['question'],
                    answer=answer,
                    metadata={
                        'full_answer': item['answer']
                    }
                )
                samples.append(sample)

        # 采样
        if max_samples is not None and max_samples < len(samples):
            random.seed(seed)
            samples = random.sample(samples, max_samples)

        return samples

    def _extract_answer(self, text: str) -> str:
        """从answer字段提取####后的数字"""
        # GSM8K格式：解题过程\n#### 答案
        if '####' in text:
            answer_part = text.split('####')[1].strip()
        else:
            answer_part = text.strip()

        # 提取第一个数字（包括负数和小数）
        match = re.search(r'-?\d+\.?\d*', answer_part)
        if match:
            return match.group(0)

        return answer_part
