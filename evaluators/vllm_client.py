"""
vLLM Python API Client
直接使用 vLLM 的 Python API，避免 HTTP 请求
"""

import os
import threading
from typing import Optional, List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VLLMClient:
    """vLLM Python API 客户端"""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
        gpu_ids: str = "0",
        tensor_parallel_size: int = None,
    ):
        """
        初始化 vLLM 客户端

        Args:
            model_path: 模型路径
            gpu_memory_utilization: GPU 内存利用率
            max_model_len: 最大模型长度
            trust_remote_code: 是否信任远程代码
            gpu_ids: GPU ID（例如 "0" 或 "0,1,2,3"）
            tensor_parallel_size: Tensor 并行大小（如果为None，自动根据gpu_ids数量设置）
        """
        self.model_path = model_path

        # 解析 GPU IDs
        # 如果已经设置了 CUDA_VISIBLE_DEVICES，使用现有设置
        existing_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_cuda_devices:
            gpu_list = [gpu.strip() for gpu in existing_cuda_devices.split(",")]
            num_gpus = len(gpu_list)
        else:
            gpu_list = [gpu.strip() for gpu in gpu_ids.split(",")]
            num_gpus = len(gpu_list)
            # 只有在未设置时才设置 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

        # 如果没有指定 tensor_parallel_size，根据 GPU 数量自动设置
        if tensor_parallel_size is None:
            tensor_parallel_size = num_gpus

        print(f"正在加载模型: {model_path}")
        print(f"使用 GPU: {gpu_ids} ({num_gpus} 个GPU)")
        print(f"Tensor 并行大小: {tensor_parallel_size}")

        # 加载 tokenizer（用于 chat template）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.has_chat_template = self.tokenizer.chat_template is not None
        if self.has_chat_template:
            print(f"检测到 chat template，将使用 ChatML 格式")
        else:
            print(f"未检测到 chat template，将使用原始格式")

        # 初始化 vLLM
        # 优化参数以更好利用GPU（H200 单卡 143GB 显存，拉满利用率）
        self.llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=65536,  # H200 显存充足，增大批量
            max_num_seqs=2048,  # 增大并发序列数
            enable_chunked_prefill=True,  # 启用chunked prefill以提升吞吐
        )

        # 添加锁来保护 vLLM 调用（vLLM 不是线程安全的）
        self._lock = threading.Lock()
        # 批量处理队列
        self._batch_queue = []
        self._batch_lock = threading.Lock()

        print("模型加载完成！")

    def _apply_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        """
        使用 tokenizer 的 chat template 格式化 prompt

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示

        Returns:
            str: 格式化后的 prompt
        """
        if not self.has_chat_template:
            # 没有 chat template，使用原始拼接
            if system_prompt:
                return f"{system_prompt}\n\n{user_prompt}"
            return user_prompt

        # 构建 messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # 应用 chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def _apply_chat_template_batch(self, user_prompts: List[str], system_prompt: str = None) -> List[str]:
        """
        批量应用 chat template

        Args:
            user_prompts: 用户 prompt 列表
            system_prompt: 系统 prompt（所有样本共享）

        Returns:
            List[str]: 格式化后的 prompt 列表
        """
        if not self.has_chat_template:
            # 没有 chat template，使用原始拼接
            if system_prompt:
                return [f"{system_prompt}\n\n{user_prompt}" for user_prompt in user_prompts]
            return user_prompts

        formatted_prompts = []
        for user_prompt in user_prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        return formatted_prompts

    def generate_batch(
        self,
        prompts: list,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: str = None,
        **kwargs
    ) -> list:
        """
        批量生成文本（更高效）

        Args:
            prompts: user prompt 列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            system_prompt: 系统 prompt（所有样本共享）
            **kwargs: 其他参数

        Returns:
            list: 生成的文本列表
        """
        # 应用 chat template（如果有）
        formatted_prompts = self._apply_chat_template_batch(prompts, system_prompt)

        # 设置采样参数
        # 添加停止条件，避免模型生成多个问题（参考eval_gsm8k.py）
        stop_sequences = kwargs.pop('stop', ["\n\nQuestion:", "Question:"])
        # 如果有 chat template，添加 <|im_end|> 作为停止条件
        if self.has_chat_template:
            stop_sequences = stop_sequences + ["<|im_end|>"]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            **kwargs
        )

        # 使用锁保护 vLLM 调用（vLLM 不是线程安全的）
        with self._lock:
            try:
                # 批量生成
                outputs = self.llm.generate(formatted_prompts, sampling_params)
                # 如果生成了多个问题，只保留第一个问题的答案（参考eval_gsm8k.py）
                results = []
                for output in outputs:
                    generated_text = output.outputs[0].text.strip()
                    # 如果生成了多个问题，只保留第一个问题的答案
                    first_problem_idx = generated_text.find('\n\nQuestion:')
                    if first_problem_idx > 0:
                        generated_text = generated_text[:first_problem_idx].strip()
                    results.append(generated_text)
                return results
            except Exception as e:
                raise Exception(f"vLLM batch generation failed: {e}")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数

        Returns:
            str: 生成的文本
        """
        # 应用 chat template（如果有）
        prompt = self._apply_chat_template(system_prompt, user_prompt)

        # 设置采样参数
        # 添加停止条件，避免模型生成多个问题（参考eval_gsm8k.py）
        stop_sequences = kwargs.pop('stop', ["\n\nQuestion:", "Question:"])
        # 如果有 chat template，添加 <|im_end|> 作为停止条件
        if self.has_chat_template:
            stop_sequences = stop_sequences + ["<|im_end|>"]
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
            **kwargs
        )

        # 使用锁保护 vLLM 调用（vLLM 不是线程安全的）
        # 注意：这会导致串行执行，但可以避免死锁
        with self._lock:
            try:
                # 生成
                outputs = self.llm.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text.strip()
                # 如果生成了多个问题，只保留第一个问题的答案（参考eval_gsm8k.py）
                first_problem_idx = generated_text.find('\n\nQuestion:')
                if first_problem_idx > 0:
                    generated_text = generated_text[:first_problem_idx].strip()
                return generated_text
            except Exception as e:
                # 如果出错，返回错误信息
                raise Exception(f"vLLM generation failed: {e}")
    
    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Chat completion 接口（兼容 OpenAI 格式）
        
        Args:
            messages: 消息列表 [{"role": "system", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        # 将 messages 转换为 prompt
        system_prompt = ""
        user_prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                if user_prompt:
                    user_prompt += "\n" + content
                else:
                    user_prompt = content
            elif role == "assistant":
                # 对于 assistant 消息，可以添加到 prompt 中
                if user_prompt:
                    user_prompt += f"\nAssistant: {content}"
        
        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

