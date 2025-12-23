"""
统一的数学答案验证器
使用 math-verify 进行答案提取和验证
"""

import re
import requests
import time
from typing import Optional, Tuple, List, Any
from ..base import BaseAnswerValidator

try:
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math-verify not installed. Install with: pip install math-verify")


class MathVerifyValidator(BaseAnswerValidator):
    """
    统一的数学答案验证器

    验证流程:
    1. 使用 math-verify 从完整解答中提取答案
    2. 使用 math-verify 进行符号数学验证
    3. 如果验证失败，使用 fallback 比较
    """

    # 默认提取配置
    DEFAULT_EXTRACTION_CONFIG = [
        LatexExtractionConfig(try_extract_without_anchor=True),
        ExprExtractionConfig(try_extract_without_anchor=True),
    ]

    # 包含字符串提取的配置（用于选择题等）
    STRING_EXTRACTION_CONFIG = [
        LatexExtractionConfig(try_extract_without_anchor=True),
        ExprExtractionConfig(try_extract_without_anchor=True),
        StringExtractionConfig(),
    ]

    def __init__(
        self,
        use_secondary_verification: bool = False,
        use_math_verify_extraction: bool = True,  # 使用 math-verify 提取
        api_key: str = "EMPTY",
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        初始化验证器

        Args:
            use_secondary_verification: 是否使用二次验证（暂未启用）
            use_math_verify_extraction: 是否使用 math-verify 进行答案提取
            api_key: API 密钥
            max_retries: API 调用最大重试次数
            timeout: API 调用超时时间
        """
        if not MATH_VERIFY_AVAILABLE:
            raise ImportError(
                "math-verify library is required. "
                "Install it with: pip install math-verify"
            )

        self.use_secondary_verification = use_secondary_verification
        self.use_math_verify_extraction = use_math_verify_extraction
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        # 用于存储问题和完整解答
        self._current_problem = None
        self._current_generation = None

        # 缓存提取结果
        self._extracted_answer = None

    def set_context(self, problem: str, generation: str):
        """
        设置当前问题和模型生成的完整解答

        Args:
            problem: 原始问题
            generation: 模型生成的完整解答
        """
        self._current_problem = problem
        self._current_generation = generation
        self._extracted_answer = None  # 重置缓存

    def extract_answer(self, text: str) -> Tuple[List[Any], str]:
        """
        使用 math-verify 从文本中提取答案

        Args:
            text: 模型生成的完整解答

        Returns:
            Tuple[List[Any], str]: (解析后的答案列表, 原始提取的字符串)
        """
        if not text:
            return [], ""

        try:
            # 使用 math-verify 提取
            parsed = parse(
                text,
                extraction_config=self.DEFAULT_EXTRACTION_CONFIG,
                extraction_mode="first_match"
            )

            # 如果没有提取到，尝试包含字符串的配置
            if not parsed:
                parsed = parse(
                    text,
                    extraction_config=self.STRING_EXTRACTION_CONFIG,
                    extraction_mode="first_match"
                )

            # 获取原始字符串表示
            if parsed:
                # 取最后一个非字符串结果，或最后一个结果
                for item in reversed(parsed):
                    if not isinstance(item, str):
                        result_str = str(item)
                        # 处理百分号：如果结果是 n*(1/100) 格式，转换为 n
                        if '*(1/100)' in result_str or '/100' in result_str:
                            # 尝试提取原始数字（从解答中找 n% 格式）
                            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
                            if percent_match:
                                result_str = percent_match.group(1)
                                # 更新 parsed 为纯数字
                                try:
                                    parsed = parse(f"${result_str}$")
                                except:
                                    pass
                        return parsed, result_str
                return parsed, str(parsed[-1]) if parsed else ""

            return [], ""

        except Exception as e:
            return [], ""

    def validate(self, predicted: str, reference: str) -> bool:
        """
        验证预测答案和参考答案是否相等

        Args:
            predicted: 提取的预测答案（如果 use_math_verify_extraction=True，此参数可能被忽略）
            reference: 参考答案

        Returns:
            bool: 是否相等
        """
        if not reference:
            return False

        # 如果启用 math-verify 提取，从完整解答中重新提取
        if self.use_math_verify_extraction and self._current_generation:
            pred_parsed, pred_str = self.extract_answer(self._current_generation)
            self._extracted_answer = pred_str  # 缓存提取结果
        else:
            pred_parsed = None
            pred_str = predicted

        # 解析参考答案
        try:
            ref_parsed = parse(f"${reference}$")
        except:
            ref_parsed = None

        # 方法1: 使用 math-verify 验证
        if pred_parsed and ref_parsed:
            try:
                if verify(pred_parsed, ref_parsed):
                    return True
            except:
                pass

        # 方法2: 如果 math-verify 失败，尝试直接解析预测答案字符串
        if pred_str and ref_parsed:
            try:
                pred_parsed_direct = parse(f"${pred_str}$")
                if pred_parsed_direct and verify(pred_parsed_direct, ref_parsed):
                    return True
            except:
                pass

        # 方法3: Fallback 比较
        return self._fallback_comparison(pred_str or predicted, reference)

    def get_extracted_answer(self) -> str:
        """获取最近一次提取的答案字符串"""
        return self._extracted_answer or ""

    def _verify_with_math_verify(self, predicted: str, reference: str) -> Tuple[bool, bool]:
        """
        使用 math-verify 验证答案

        Args:
            predicted: 预测答案
            reference: 参考答案

        Returns:
            Tuple[bool, bool]: (验证结果, 验证是否成功执行)
        """
        try:
            # 确保答案被 LaTeX 环境包围
            prediction_with_env = f"${predicted}$"
            reference_with_env = f"${reference}$"

            # 使用 math-verify 解析和验证
            parsed_prediction = parse(prediction_with_env)
            parsed_reference = parse(reference_with_env)

            is_correct = verify(parsed_prediction, parsed_reference)
            return is_correct, True

        except Exception as e:
            # math-verify 解析失败
            return False, False

    def _verify_with_deepseek(
        self,
        problem: str,
        reference: str,
        generation: str
    ) -> Optional[bool]:
        """
        使用 DeepSeek-V3 进行二次验证

        Args:
            problem: 原始问题
            reference: 参考答案
            generation: 模型生成的完整解答

        Returns:
            Optional[bool]: 验证结果，None 表示验证失败或 AMBIGUOUS
        """
        prompt = self.VERIFICATION_PROMPT_TEMPLATE.format(
            problem=problem,
            answer=reference,
            generation=generation
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.DEEPSEEK_API_URL,
                    json={
                        "model": self.DEEPSEEK_MODEL,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 512
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    return self._parse_verdict(content)
                elif response.status_code == 429:
                    # Rate limit
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def _parse_verdict(self, response: str) -> Optional[bool]:
        """
        解析 DeepSeek-V3 的判决结果

        Args:
            response: API 响应内容

        Returns:
            Optional[bool]: True=EQUIVALENT, False=DIFFERENT, None=AMBIGUOUS
        """
        response_upper = response.upper()

        if "VERDICT: EQUIVALENT" in response_upper:
            return True
        elif "VERDICT: DIFFERENT" in response_upper:
            return False
        elif "VERDICT: AMBIGUOUS" in response_upper:
            return None

        # 如果没有找到标准格式，尝试其他匹配
        if "EQUIVALENT" in response_upper and "DIFFERENT" not in response_upper:
            return True
        elif "DIFFERENT" in response_upper and "EQUIVALENT" not in response_upper:
            return False

        return None

    def _fallback_comparison(self, predicted: str, reference: str) -> bool:
        """
        Fallback 比较方法（当 math-verify 和二次验证都失败时）

        Args:
            predicted: 预测答案
            reference: 参考答案

        Returns:
            bool: 是否相等
        """
        # 清理字符串
        pred_clean = self._normalize(predicted)
        ref_clean = self._normalize(reference)

        # 字符串完全匹配
        if pred_clean == ref_clean:
            return True

        # 尝试数值比较
        try:
            pred_num = float(pred_clean.replace(',', ''))
            ref_num = float(ref_clean.replace(',', ''))
            return abs(pred_num - ref_num) < 1e-9
        except (ValueError, TypeError):
            pass

        return False

    def _normalize(self, text: str) -> str:
        """规范化答案字符串"""
        if not text:
            return ""

        text = text.strip().lower()

        # 移除 LaTeX 符号
        text = text.replace('$', '')
        text = text.replace('\\', '')

        # 移除空格
        text = re.sub(r'\s+', '', text)

        return text
