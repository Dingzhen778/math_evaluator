"""
Main Evaluator
统一的评测流程，支持MATH500, GSM8K, AIME24, AIME25
"""

import json
import threading
from typing import List, Dict, Any, Optional, Callable, Any as AnyType
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base import EvalSample, EvalResult, BaseDatasetLoader, BaseAnswerExtractor, BaseAnswerValidator, BasePromptBuilder
from .datasets import MATHDatasetLoader, GSM8KDatasetLoader, AIME2024DatasetLoader, AIME2025DatasetLoader, GaoKao2023ENDatasetLoader, MathOdysseyDatasetLoader, AMC23DatasetLoader, OlympiadBenchOEDatasetLoader, OlympiadBenchTPDatasetLoader
from .extractors import MATHAnswerExtractor, GSM8KAnswerExtractor, AImeAnswerExtractor
from .validators import MATHAnswerValidator, NumericAnswerValidator, AIMEAnswerValidator, MathVerifyValidator
from .prompts import MATHPromptBuilder, GSM8KPromptBuilder, AImePromptBuilder


class MathEvaluator:
    """
    数学评测框架主类

    支持的数据集：
    - MATH500: MATH数据集的500个样本子集
    - GSM8K: Grade School Math 8K
    - AIME2024: AIME 2024竞赛题
    - AIME2025: AIME 2025竞赛题
    """

    DATASET_CONFIGS = {
        'math': {
            'loader': MATHDatasetLoader,
            'extractor': MATHAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': MATHPromptBuilder,
        },
        'math500': {
            'loader': MATHDatasetLoader,
            'extractor': MATHAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': MATHPromptBuilder,
        },
        'gsm8k': {
            'loader': GSM8KDatasetLoader,
            'extractor': GSM8KAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': GSM8KPromptBuilder,
        },
        'aime2024': {
            'loader': AIME2024DatasetLoader,
            'extractor': AImeAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': AImePromptBuilder,
        },
        'aime2025': {
            'loader': AIME2025DatasetLoader,
            'extractor': AImeAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': AImePromptBuilder,
        },
        'gaokao2023en': {
            'loader': GaoKao2023ENDatasetLoader,
            'extractor': MATHAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': MATHPromptBuilder,
        },
        'mathodyssey': {
            'loader': MathOdysseyDatasetLoader,
            'extractor': MATHAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': MATHPromptBuilder,
        },
        'amc23': {
            'loader': AMC23DatasetLoader,
            'extractor': AImeAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': AImePromptBuilder,
        },
        'olympiadbench_oe': {
            'loader': OlympiadBenchOEDatasetLoader,
            'extractor': MATHAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': MATHPromptBuilder,
        },
        'olympiadbench_tp': {
            'loader': OlympiadBenchTPDatasetLoader,
            'extractor': MATHAnswerExtractor,
            'validator': MathVerifyValidator,
            'prompt_builder': MATHPromptBuilder,
        },
    }

    def __init__(
        self,
        dataset_name: str,
        model_generate_fn: Optional[Callable] = None,
        max_workers: int = 32,
        use_few_shot: bool = True,
        vllm_client: Optional[Any] = None,
        answer_format: str = "auto",
    ):
        """
        初始化评测器

        Args:
            dataset_name: 数据集名称 ('math', 'math500', 'gsm8k', 'aime2024', 'aime2025')
            model_generate_fn: 模型生成函数，签名为 fn(system_prompt: str, user_prompt: str) -> str
                             如果为None，则需要在evaluate时提供
            max_workers: 并行评测的线程数
            use_few_shot: 是否使用few-shot示例
            vllm_client: vLLM客户端对象（如果提供，将使用批量处理）
            answer_format: 答案格式 ("auto", "boxed", "hash")
                          - "auto": 自动检测，所有数据集默认使用 boxed 格式
                          - "boxed": 使用 \\boxed{} 格式 (适用于 Qwen-Math 等模型)
                          - "hash": 使用 #### 格式 (适用于部分 LLaMA 模型)
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_CONFIGS.keys())}")

        self.dataset_name = dataset_name
        self.model_generate_fn = model_generate_fn
        self.vllm_client = vllm_client
        self.max_workers = max_workers
        self.use_few_shot = use_few_shot
        self.answer_format = answer_format

        # 初始化组件
        config = self.DATASET_CONFIGS[dataset_name]
        self.loader: BaseDatasetLoader = config['loader']()
        self.extractor: BaseAnswerExtractor = config['extractor']()
        self.validator: BaseAnswerValidator = config['validator']()

        # 根据 answer_format 初始化 prompt_builder
        prompt_builder_class = config['prompt_builder']
        if prompt_builder_class == GSM8KPromptBuilder:
            # GSM8K 支持两种格式
            fmt = "boxed" if answer_format in ("auto", "boxed") else "hash"
            self.prompt_builder: BasePromptBuilder = prompt_builder_class(answer_format=fmt)
        else:
            # 其他数据集默认使用 boxed 格式
            self.prompt_builder: BasePromptBuilder = prompt_builder_class()

        # 线程安全的计数器
        self._lock = threading.Lock()
        self._correct_count = 0
        self._total_count = 0

    def load_dataset(
        self,
        path: Optional[str] = None,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> List[EvalSample]:
        """
        加载数据集

        Args:
            path: 数据集路径（如果为None，使用默认路径）
            max_samples: 最大样本数（用于采样）
            **kwargs: 传递给loader的其他参数

        Returns:
            List[EvalSample]: 评测样本列表
        """
        # 特殊处理MATH500
        if self.dataset_name == 'math500':
            if max_samples is None:
                max_samples = 500

        return self.loader.load(path=path, max_samples=max_samples, **kwargs)

    def evaluate_single(
        self,
        sample: EvalSample,
        model_generate_fn: Optional[Callable] = None,
    ) -> EvalResult:
        """
        评测单个样本

        Args:
            sample: 评测样本
            model_generate_fn: 模型生成函数（覆盖初始化时的函数）

        Returns:
            EvalResult: 评测结果
        """
        # 使用提供的函数或初始化时的函数
        generate_fn = model_generate_fn or self.model_generate_fn
        if generate_fn is None:
            raise ValueError("model_generate_fn must be provided either in __init__ or in evaluate_single")

        # 构建prompt
        system_prompt = self.prompt_builder.build_system_prompt()

        # 获取few-shot示例
        few_shot_examples = None
        if self.use_few_shot and hasattr(self.prompt_builder, 'get_few_shot_examples'):
            few_shot_examples = self.prompt_builder.get_few_shot_examples()

        user_prompt = self.prompt_builder.build_user_prompt(
            question=sample.question,
            few_shot_examples=few_shot_examples
        )

        # 调用模型生成
        generated_solution = generate_fn(system_prompt, user_prompt)

        # 设置验证器上下文（math-verify 会从中提取答案）
        if hasattr(self.validator, 'set_context'):
            self.validator.set_context(
                problem=sample.question,
                generation=generated_solution
            )

        # 提取答案（先用旧提取器作为 fallback）
        predicted_answer = self.extractor.extract(generated_solution)

        # 验证答案（验证器内部会用 math-verify 重新提取）
        is_correct = self.validator.validate(predicted_answer, sample.answer)

        # 获取验证器提取的答案（如果有）
        if hasattr(self.validator, 'get_extracted_answer'):
            extracted = self.validator.get_extracted_answer()
            if extracted:
                predicted_answer = extracted

        # 更新计数（线程安全）
        with self._lock:
            if is_correct:
                self._correct_count += 1
            self._total_count += 1

        return EvalResult(
            question=sample.question,
            reference_answer=sample.answer,
            predicted_answer=predicted_answer,
            generated_solution=generated_solution,
            is_correct=is_correct,
            metadata=sample.metadata
        )

    def evaluate(
        self,
        samples: List[EvalSample],
        model_generate_fn: Optional[Callable] = None,
        show_progress: bool = True,
        batch_size: int = 512,  # H200 单卡显存充足，增大 batch
    ) -> Dict[str, Any]:
        """
        评测整个数据集

        Args:
            samples: 评测样本列表
            model_generate_fn: 模型生成函数（覆盖初始化时的函数）
            show_progress: 是否显示进度条
            batch_size: 批量处理大小（如果模型支持批量处理）

        Returns:
            Dict: 评测结果
                - accuracy: 准确率
                - correct: 正确数量
                - total: 总数量
                - results: 详细结果列表
        """
        # 重置计数器
        self._correct_count = 0
        self._total_count = 0

        results = []
        
        # 检查是否支持批量处理
        # 优先使用传入的 vllm_client，否则检查 model_generate_fn
        vllm_client = None
        if hasattr(self, 'vllm_client') and self.vllm_client is not None:
            vllm_client = self.vllm_client
        elif model_generate_fn and hasattr(model_generate_fn, '__self__'):
            potential_client = model_generate_fn.__self__
            if hasattr(potential_client, 'generate_batch'):
                vllm_client = potential_client
        
        # 如果找到 vllm_client，使用批量处理
        if vllm_client and hasattr(vllm_client, 'generate_batch'):
            return self._evaluate_batch(samples, vllm_client, show_progress, batch_size)
        
        # 否则使用原来的并行处理
        generate_fn = model_generate_fn or self.model_generate_fn
        if generate_fn is None:
            raise ValueError("model_generate_fn must be provided")
        
        # 否则使用原来的并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.evaluate_single, sample, model_generate_fn): sample
                for sample in samples
            }

            # 使用tqdm显示进度
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(samples), desc=f"Evaluating {self.dataset_name}")

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)

                    # 更新进度条描述
                    if show_progress and self._total_count > 0:
                        accuracy = self._correct_count / self._total_count * 100
                        iterator.set_postfix({'accuracy': f'{accuracy:.2f}%'})
                except Exception as e:
                    sample = futures[future]
                    print(f"Error evaluating sample: {sample.question[:50]}... Error: {e}")

        # 计算最终准确率
        accuracy = self._correct_count / self._total_count * 100 if self._total_count > 0 else 0

        return {
            'dataset': self.dataset_name,
            'accuracy': round(accuracy, 2),
            'correct': self._correct_count,
            'total': self._total_count,
            'results': [self._result_to_dict(r) for r in results]
        }
    
    def _evaluate_batch(
        self,
        samples: List[EvalSample],
        vllm_client: Any,
        show_progress: bool = True,
        batch_size: int = 512,  # H200 单卡显存充足，增大 batch
    ) -> Dict[str, Any]:
        """批量处理评测（更高效）"""
        results = []

        # 构建所有 prompts
        system_prompt = self.prompt_builder.build_system_prompt()
        few_shot_examples = None
        if self.use_few_shot and hasattr(self.prompt_builder, 'get_few_shot_examples'):
            few_shot_examples = self.prompt_builder.get_few_shot_examples()

        user_prompts = []
        for sample in samples:
            user_prompt = self.prompt_builder.build_user_prompt(
                question=sample.question,
                few_shot_examples=few_shot_examples
            )
            user_prompts.append(user_prompt)

        # 批量处理
        num_batches = (len(samples) + batch_size - 1) // batch_size
        iterator = range(0, len(samples), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Evaluating {self.dataset_name}", total=num_batches)

        for i in iterator:
            batch_samples = samples[i:i+batch_size]
            batch_user_prompts = user_prompts[i:i+batch_size]

            # 批量生成（传递 system_prompt 和 user_prompts 分开）
            try:
                generated_solutions = vllm_client.generate_batch(
                    batch_user_prompts,
                    system_prompt=system_prompt
                )
                
                # 处理每个结果
                for sample, generated_solution in zip(batch_samples, generated_solutions):
                    # 设置验证器上下文（math-verify 会从中提取答案）
                    if hasattr(self.validator, 'set_context'):
                        self.validator.set_context(
                            problem=sample.question,
                            generation=generated_solution
                        )

                    # 提取答案（先用旧提取器作为 fallback）
                    predicted_answer = self.extractor.extract(generated_solution)

                    # 验证答案（验证器内部会用 math-verify 重新提取）
                    is_correct = self.validator.validate(predicted_answer, sample.answer)

                    # 获取验证器提取的答案（如果有）
                    if hasattr(self.validator, 'get_extracted_answer'):
                        extracted = self.validator.get_extracted_answer()
                        if extracted:
                            predicted_answer = extracted

                    # 更新计数
                    with self._lock:
                        if is_correct:
                            self._correct_count += 1
                        self._total_count += 1

                    results.append(EvalResult(
                        question=sample.question,
                        reference_answer=sample.answer,
                        predicted_answer=predicted_answer,
                        generated_solution=generated_solution,
                        is_correct=is_correct,
                        metadata=sample.metadata
                    ))
                    
                    # 更新进度条
                    if show_progress and self._total_count > 0:
                        accuracy = self._correct_count / self._total_count * 100
                        iterator.set_postfix({'accuracy': f'{accuracy:.2f}%', 'processed': f'{self._total_count}/{len(samples)}'})
                        
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # 如果批量失败，回退到单个处理
                for sample in batch_samples:
                    try:
                        result = self.evaluate_single(sample, lambda sp, up: vllm_client.generate(sp, up))
                        results.append(result)
                    except Exception as e2:
                        print(f"Error evaluating sample: {sample.question[:50]}... Error: {e2}")
        
        # 计算最终准确率
        accuracy = self._correct_count / self._total_count * 100 if self._total_count > 0 else 0
        
        return {
            'dataset': self.dataset_name,
            'accuracy': round(accuracy, 2),
            'correct': self._correct_count,
            'total': self._total_count,
            'results': [self._result_to_dict(r) for r in results]
        }

    def evaluate_and_save(
        self,
        output_path: str,
        dataset_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        model_generate_fn: Optional[Callable] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        完整的评测流程：加载数据集 -> 评测 -> 保存结果

        Args:
            output_path: 结果保存路径（JSON文件）
            dataset_path: 数据集路径
            max_samples: 最大样本数
            model_generate_fn: 模型生成函数
            model_name: 模型名称（会保存到结果中）
            **kwargs: 传递给load_dataset的其他参数
        """
        # 加载数据集
        print(f"Loading {self.dataset_name} dataset...")
        samples = self.load_dataset(path=dataset_path, max_samples=max_samples, **kwargs)
        print(f"Loaded {len(samples)} samples")

        # 评测
        print(f"Evaluating {self.dataset_name}...")
        results = self.evaluate(samples, model_generate_fn=model_generate_fn)

        # 添加模型信息到结果中
        if model_name:
            results['model'] = model_name

        # 保存结果
        print(f"Saving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed!")
        print(f"Model: {model_name or 'N/A'}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Correct: {results['correct']}/{results['total']}")

        return results

    @staticmethod
    def _result_to_dict(result: EvalResult) -> Dict[str, Any]:
        """将EvalResult转换为字典"""
        return {
            'question': result.question,
            'reference_answer': result.reference_answer,
            'predicted_answer': result.predicted_answer,
            'generated_solution': result.generated_solution,
            'is_correct': result.is_correct,
            'metadata': result.metadata
        }
