"""
新增数据集加载器: GaoKao2023EN, MathOdyssey, AMC23
"""

import json
import os
from typing import List, Optional
from ..base import BaseDatasetLoader, EvalSample


class GaoKao2023ENDatasetLoader(BaseDatasetLoader):
    """GaoKao2023EN 数据集加载器 (385条)"""

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'gaokao2023en', 'gaokao2023en.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample = EvalSample(
                    question=item.get('question', ''),
                    answer=str(item.get('answer', '')),
                    metadata={'source': 'GaoKao2023EN'}
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
        return samples


class MathOdysseyDatasetLoader(BaseDatasetLoader):
    """MathOdyssey 数据集加载器 (389条)"""

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'mathodyssey', 'odyssey_test.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample = EvalSample(
                    question=item.get('problem_statement', ''),
                    answer=str(item.get('answer', '')),
                    metadata={'source': 'MathOdyssey', 'label': item.get('label', '')}
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
        return samples


class AMC23DatasetLoader(BaseDatasetLoader):
    """AMC23 数据集加载器 (40条)"""

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'amc23', 'amc23_test.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample = EvalSample(
                    question=item.get('question', ''),
                    answer=str(item.get('answer', '')),
                    metadata={'source': 'AMC23', 'url': item.get('url', '')}
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
        return samples


class OlympiadBenchOEDatasetLoader(BaseDatasetLoader):
    """OlympiadBench OE_TO 开放式纯文本数据集加载器 (675条)"""

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'olympiadbench', 'OE_TO_maths_en.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 跳过没有答案的题目
                final_answer = item.get('final_answer')
                if final_answer is None:
                    continue
                # 答案可能是列表，取第一个
                if isinstance(final_answer, list):
                    answer = str(final_answer[0]) if final_answer else ''
                else:
                    answer = str(final_answer)

                sample = EvalSample(
                    question=item.get('question', ''),
                    answer=answer,
                    metadata={
                        'source': 'OlympiadBench_OE',
                        'subfield': item.get('subfield', ''),
                        'question_id': item.get('question_id', '')
                    }
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
        return samples


class OlympiadBenchTPDatasetLoader(BaseDatasetLoader):
    """OlympiadBench TP_TO 证明题纯文本数据集加载器 (503条)

    注意：证明题没有标准答案，评测时需要特殊处理
    """

    def load(self, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[EvalSample]:
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'olympiadbench', 'TP_TO_maths_en.jsonl')

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                sample = EvalSample(
                    question=item.get('question', ''),
                    answer='PROOF',  # 证明题标记
                    metadata={
                        'source': 'OlympiadBench_TP',
                        'subfield': item.get('subfield', ''),
                        'question_id': item.get('question_id', '')
                    }
                )
                samples.append(sample)

        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
        return samples
