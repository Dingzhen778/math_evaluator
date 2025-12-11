"""
Base classes for evaluation framework
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EvalSample:
    """单个评测样本"""
    question: str
    answer: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EvalResult:
    """单个样本的评测结果"""
    question: str
    reference_answer: str
    predicted_answer: str
    generated_solution: str
    is_correct: bool
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDatasetLoader(ABC):
    """数据集加载器基类"""

    @abstractmethod
    def load(self, path: str, **kwargs) -> List[EvalSample]:
        """加载数据集"""
        pass


class BaseAnswerExtractor(ABC):
    """答案提取器基类"""

    @abstractmethod
    def extract(self, text: str) -> str:
        """从模型输出中提取答案"""
        pass


class BaseAnswerValidator(ABC):
    """答案验证器基类"""

    @abstractmethod
    def validate(self, predicted: str, reference: str) -> bool:
        """验证答案是否正确"""
        pass


class BasePromptBuilder(ABC):
    """Prompt构建器基类（针对instruct模型）"""

    @abstractmethod
    def build_system_prompt(self) -> str:
        """构建系统提示"""
        pass

    @abstractmethod
    def build_user_prompt(self, question: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """构建用户提示"""
        pass
