"""
AIME Dataset Loaders (AIME 2024 & 2025)
数据集从本地JSONL文件加载（下载自HuggingFace）
"""

import json
import os
from typing import List, Optional
from ..base import BaseDatasetLoader, EvalSample


class AIME2024DatasetLoader(BaseDatasetLoader):
    """AIME 2024数据集加载器

    数据源: Maxwell-Jia/AIME_2024 on HuggingFace
    样本数: 30 (AIME 2024 完整题目)
    """

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        """
        加载AIME 2024数据集

        Args:
            path: JSONL文件路径（可选，默认使用相对路径）
            max_samples: 最大样本数

        Returns:
            List[EvalSample]: 评测样本列表
        """
        if path is None:
            # 使用相对路径：相对于当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'aime', 'aime2024.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample = EvalSample(
                    question=item.get('question', item.get('origin_prompt', '')),
                    answer=str(item.get('answer', item.get('gold_answer', ''))),
                    metadata={
                        'year': 2024,
                        'source': 'AIME2024'
                    }
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]

        return samples


class AIME2025DatasetLoader(BaseDatasetLoader):
    """AIME 2025数据集加载器

    数据源: opencompass/AIME2025 on HuggingFace
    样本数: 30 (15 from AIME2025-I + 15 from AIME2025-II)
    """

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        """
        加载AIME 2025数据集

        Args:
            path: JSONL文件路径（可选，默认使用相对路径）
            max_samples: 最大样本数

        Returns:
            List[EvalSample]: 评测样本列表
        """
        if path is None:
            # 使用相对路径：相对于当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'aime', 'aime2025.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample = EvalSample(
                    question=item.get('question', item.get('origin_prompt', '')),
                    answer=str(item.get('answer', item.get('gold_answer', ''))),
                    metadata={
                        'year': 2025,
                        'source': 'AIME2025'
                    }
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]

        return samples
