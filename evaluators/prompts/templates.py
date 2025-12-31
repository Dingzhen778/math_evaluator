"""
Prompt Templates for Instruct Models
针对instruct模型设计的prompt模板
"""

from typing import List, Dict, Optional
from ..base import BasePromptBuilder


class MATHPromptBuilder(BasePromptBuilder):
    """MATH数据集的Prompt构建器"""

    def build_system_prompt(self) -> str:
        """构建系统提示"""
        return """You are an expert mathematician. Solve the given math problem step by step and provide your final answer in \\boxed{} format.

Guidelines:
- Show your reasoning process clearly
- Provide the final answer enclosed in \\boxed{}
- The answer should be exact (not approximated) whenever possible"""

    def build_user_prompt(self, question: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """构建用户提示"""
        prompt_parts = []

        # 添加few-shot示例（如果提供）
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Problem: {example['question']}")
                prompt_parts.append(f"Solution: {example['solution']}\n")

        # 添加当前问题
        prompt_parts.append("Problem:")
        prompt_parts.append(question)
        prompt_parts.append("\nPlease solve this problem step by step and provide your final answer in \\boxed{} format.")

        return "\n".join(prompt_parts)

    def get_few_shot_examples(self) -> List[Dict]:
        """获取默认的few-shot示例（参考opencompass和eval_base_model）"""
        return [
            {
                "question": "Find the greatest common divisor of 12 and 18.",
                "solution": "The factors of 12 are 1, 2, 3, 4, 6, and 12. The factors of 18 are 1, 2, 3, 6, 9, and 18. The common factors are 1, 2, 3, and 6. Therefore, the greatest common divisor is \\boxed{6}."
            },
            {
                "question": "Solve for x: 2x + 5 = 13",
                "solution": "Subtract 5 from both sides: 2x = 8. Divide both sides by 2: x = 4. Therefore, \\boxed{4}."
            },
            {
                "question": "What is the area of a circle with radius 5?",
                "solution": "The area of a circle is given by A = πr². With r = 5, we have A = π(5)² = 25π. Therefore, \\boxed{25\\pi}."
            },
            {
                "question": "If f(x) = x² + 2x + 1, what is f(3)?",
                "solution": "Substitute x = 3 into the function: f(3) = (3)² + 2(3) + 1 = 9 + 6 + 1 = 16. Therefore, \\boxed{16}."
            },
        ]


class GSM8KPromptBuilder(BasePromptBuilder):
    """GSM8K数据集的Prompt构建器，支持不同答案格式"""

    def __init__(self, answer_format: str = "hash"):
        """
        初始化 GSM8K Prompt 构建器

        Args:
            answer_format: 答案格式，"hash" 使用 #### 格式，"boxed" 使用 \\boxed{} 格式
        """
        self.answer_format = answer_format

    def build_system_prompt(self) -> str:
        """构建系统提示"""
        if self.answer_format == "boxed":
            return """You are an expert at solving grade school math word problems. Solve the given problem step by step and provide your final numerical answer in \\boxed{} format.

Guidelines:
- Break down the problem into clear steps
- Show your arithmetic calculations
- Provide the final answer as a number enclosed in \\boxed{}"""
        else:
            return """You are an expert at solving grade school math word problems. Solve the given problem step by step and provide your final numerical answer after ####.

Guidelines:
- Break down the problem into clear steps
- Show your arithmetic calculations
- Provide the final answer as a number after ####"""

    def build_user_prompt(self, question: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """构建用户提示"""
        prompt_parts = []

        # 添加few-shot示例（如果提供）
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples, 1):
                prompt_parts.append(f"Question: {example['question']}")
                prompt_parts.append(f"Answer: {example['solution']}\n")

        # 添加当前问题
        prompt_parts.append(f"Question: {question}")
        if self.answer_format == "boxed":
            prompt_parts.append("Answer: Let's solve this step by step.")
        else:
            prompt_parts.append("Answer: Let's think step by step.")

        return "\n".join(prompt_parts)

    def get_few_shot_examples(self) -> List[Dict]:
        """获取默认的few-shot示例"""
        if self.answer_format == "boxed":
            # 使用 \boxed{} 格式的示例（适用于 Qwen-Math 等模型）
            return [
                {
                    "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                    "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. The answer is \\boxed{6}."
                },
                {
                    "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                    "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is \\boxed{5}."
                },
                {
                    "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                    "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is \\boxed{39}."
                },
                {
                    "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                    "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is \\boxed{8}."
                },
                {
                    "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                    "solution": "Shawn started with 5 toys. He got 2 toys from mom and 2 toys from dad, that is 4 more toys. 5 + 4 = 9. The answer is \\boxed{9}."
                },
                {
                    "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                    "solution": "There were originally 9 computers. For each of 4 days (Monday to Thursday), 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is \\boxed{29}."
                },
                {
                    "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                    "solution": "Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more on Wednesday, he had 35 - 2 = 33 golf balls. The answer is \\boxed{33}."
                },
                {
                    "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                    "solution": "Olivia had 23 dollars. 5 bagels for 3 dollars each costs 5 × 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is \\boxed{8}."
                },
            ]
        else:
            # 使用 #### 格式的示例（适用于 LLaMA 等模型）
            return [
                {
                    "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                    "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6"
                },
                {
                    "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                    "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5"
                },
                {
                    "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                    "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
                },
                {
                    "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                    "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8"
                },
                {
                    "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                    "solution": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9"
                },
                {
                    "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                    "solution": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29"
                },
                {
                    "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                    "solution": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"
                },
                {
                    "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                    "solution": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8"
                },
            ]


class AImePromptBuilder(BasePromptBuilder):
    """AIME数据集的Prompt构建器（2024和2025通用）"""

    def build_system_prompt(self) -> str:
        """构建系统提示"""
        return """You are an expert mathematician specializing in competition-level mathematics. Solve the given AIME problem step by step and provide your final answer.

Guidelines:
- AIME answers are always integers from 0 to 999
- Show your complete reasoning process
- Provide the final answer as an integer enclosed in \\boxed{}
- Be precise and rigorous in your mathematical reasoning"""

    def build_user_prompt(self, question: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """构建用户提示"""
        prompt_parts = []

        # AIME通常使用zero-shot或few-shot
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Problem: {example['question']}")
                prompt_parts.append(f"Solution: {example['solution']}\n")

        # 添加当前问题
        prompt_parts.append("Problem:")
        prompt_parts.append(question)
        prompt_parts.append("\nPlease solve this problem step by step. Remember that the answer must be an integer from 0 to 999. Provide your final answer in \\boxed{} format.")

        return "\n".join(prompt_parts)

    def get_few_shot_examples(self) -> List[Dict]:
        """获取默认的few-shot示例（AIME通常使用zero-shot或1-2个示例）"""
        return [
            {
                "question": "What is the remainder when 2^2023 is divided by 7?",
                "solution": "We need to find the pattern of powers of 2 modulo 7. 2^1 = 2, 2^2 = 4, 2^3 = 8 ≡ 1 (mod 7). So the cycle length is 3. Since 2023 = 3 × 674 + 1, we have 2^2023 ≡ 2^1 ≡ 2 (mod 7). Therefore, \\boxed{2}."
            },
        ]
