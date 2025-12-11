"""
API Client for Model Inference
支持OpenAI兼容的API调用
"""

import os
from typing import Optional, Dict, Any
import requests
import time


class APIClient:
    """OpenAI兼容的API客户端"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        timeout: int = 120,
    ):
        """
        初始化API客户端

        Args:
            base_url: API基础URL（例如：https://api.openai.com/v1）
            model: 模型名称
            api_key: API密钥
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        调用chat completion API

        Args:
            messages: 消息列表 [{"role": "system", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大生成token数
            **kwargs: 其他参数

        Returns:
            str: 模型生成的文本
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                elif response.status_code == 429:
                    # Rate limit, wait and retry
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"Timeout, retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Request error: {e}, retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise

        raise Exception(f"Failed after {self.max_retries} retries")

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        便捷方法：生成文本

        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            **kwargs: 传递给chat_completion的其他参数

        Returns:
            str: 模型生成的文本
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.chat_completion(messages, **kwargs)


def create_api_client_from_env() -> APIClient:
    """
    从环境变量创建API客户端

    环境变量：
    - EVAL_BASE_URL: API基础URL
    - EVAL_MODEL: 模型名称
    - EVAL_API_KEY: API密钥（可选，默认EMPTY）
    """
    base_url = os.environ.get('EVAL_BASE_URL')
    model = os.environ.get('EVAL_MODEL')
    api_key = os.environ.get('EVAL_API_KEY', 'EMPTY')

    if not base_url or not model:
        raise ValueError(
            "Please set EVAL_BASE_URL and EVAL_MODEL environment variables"
        )

    return APIClient(base_url=base_url, model=model, api_key=api_key)
