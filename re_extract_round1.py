#!/usr/bin/env python3
"""
重新提取round1结果的答案，使用改进后的提取逻辑
"""
import json
import re
import os
from pathlib import Path
from typing import Optional

def extract_answer_improved(text: str) -> Optional[str]:
    """
    兼容多种提取方法的答案提取逻辑
    尝试所有方法，如果多种方法提取到相同答案则更可信
    """
    """
    改进的答案提取逻辑（参考eval_gsm8k.py）
    
    优先级：
    1. "The answer is" 模式（few-shot示例中的格式）
    2. #### 后的数字（优先取第一个，避免提取到重复示例的答案）
    3. 最后一个数字（先移除"Question:"之后的内容，避免提取到新问题）
    """
    if not text or not text.strip():
        return None
    
    # 收集所有可能的答案（去重）
    candidates = []
    
    # 方法1: 查找"The answer is"模式（few-shot示例中的格式，优先级最高）
    answer_patterns = [
        r'The answer is\s+([\d,]+\.?\d*)',
        r'the answer is\s+([\d,]+\.?\d*)',
        r'answer is\s+([\d,]+\.?\d*)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().replace(',', '').rstrip('.')
            if answer:
                candidates.append(('answer_is', answer))

    # 方法2: 提取#### 后的数字（尝试第一个和第二个）
    if '####' in text:
        pattern = r'####\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)'
        matches = list(re.finditer(pattern, text))
        if matches:
            first_match = matches[0]
            first_pos = first_match.start()
            if len(matches) > 1 and first_pos < len(text) * 0.2:
                answer = matches[1].group(1).replace(',', '')
            else:
                answer = first_match.group(1).replace(',', '')
            if answer:
                candidates.append(('hash_marker', answer))

    # 方法3: 提取\\boxed{}
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    boxed_matches = list(re.finditer(boxed_pattern, text))
    if boxed_matches:
        last_match = boxed_matches[-1]
        content = last_match.group(1)
        match = re.search(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', content)
        if match:
            answer = match.group(0).replace(',', '')
            if answer:
                candidates.append(('boxed', answer))

    # 方法4: 提取最后一个数字（先移除"Question:"之后的内容）
    text_before_question = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text_before_question)
    if numbers:
        candidates.append(('last_number', numbers[-1].replace(',', '')))
    
    if not candidates:
        return None

    # 标准化答案并统计
    def normalize_answer(answer: str) -> Optional[str]:
        if not answer:
            return None
        normalized = answer.replace(',', '').rstrip('.')
        try:
            if '.' in normalized:
                num = float(normalized)
                if num.is_integer():
                    return str(int(num))
                return normalized
            return normalized
        except (ValueError, TypeError):
            return normalized

    answer_counts = {}
    answer_priority = {}
    priority_map = {'answer_is': 1, 'hash_marker': 2, 'boxed': 3, 'last_number': 4}
    
    for method, answer in candidates:
        normalized = normalize_answer(answer)
        if normalized:
            if normalized not in answer_counts:
                answer_counts[normalized] = 0
                answer_priority[normalized] = priority_map.get(method, 999)
            answer_counts[normalized] += 1
            answer_priority[normalized] = min(answer_priority[normalized], priority_map.get(method, 999))

    # 选择最佳答案：优先选择出现次数最多的，如果次数相同则选择优先级最高的
    if answer_counts:
        max_count = max(answer_counts.values())
        best_answers = [ans for ans, count in answer_counts.items() if count == max_count]
        if len(best_answers) == 1:
            return best_answers[0]
        else:
            return min(best_answers, key=lambda x: answer_priority[x])
    
    return None

def is_equal(pred: str, refer: str) -> bool:
    """判断两个答案是否相等（参考OpenCompass的Gsm8kEvaluator）"""
    if not pred or not refer:
        return False
    try:
        # 直接比较
        if pred == refer:
            return True
        # 数值比较（允许浮点数误差）
        pred_float = float(pred)
        refer_float = float(refer)
        if abs(pred_float - refer_float) < 1e-6:
            return True
        # 如果参考答案是整数，预测答案的浮点数应该接近整数
        if abs(pred_float - int(refer_float)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return False

def re_extract_results(round_dir: str):
    """重新提取指定目录下所有GSM8K结果文件的答案"""
    round_path = Path(round_dir)
    if not round_path.exists():
        print(f"错误: 目录不存在: {round_dir}")
        return
    
    # 查找所有GSM8K结果文件
    gsm8k_files = list(round_path.glob("gsm8k_*.json"))
    
    if not gsm8k_files:
        print(f"未找到GSM8K结果文件 in {round_dir}")
        return
    
    print(f"找到 {len(gsm8k_files)} 个GSM8K结果文件")
    print("=" * 80)
    
    total_improvement = 0
    total_old_correct = 0
    total_new_correct = 0
    total_samples = 0
    
    for json_file in sorted(gsm8k_files):
        print(f"\n处理文件: {json_file.name}")
        
        # 读取原始结果
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        old_correct = data.get('correct', 0)
        old_total = data.get('total', 0)
        old_accuracy = data.get('accuracy', 0.0)
        
        # 重新提取答案
        new_correct = 0
        new_total = 0
        
        for result in data.get('results', []):
            generated_solution = result.get('generated_solution', '')
            reference_answer = result.get('reference_answer', '')
            
            if not generated_solution:
                continue
            
            # 使用改进后的提取逻辑
            new_predicted_answer = extract_answer_improved(generated_solution)
            
            # 判断是否正确
            is_correct = False
            if new_predicted_answer:
                is_correct = is_equal(new_predicted_answer, reference_answer)
            
            # 更新结果
            result['predicted_answer_new'] = new_predicted_answer
            result['is_correct_new'] = is_correct
            
            new_total += 1
            if is_correct:
                new_correct += 1
        
        new_accuracy = (new_correct / new_total * 100) if new_total > 0 else 0.0
        
        # 统计
        total_old_correct += old_correct
        total_new_correct += new_correct
        total_samples += new_total
        improvement = new_correct - old_correct
        total_improvement += improvement
        
        print(f"  模型: {data.get('model', 'unknown')}")
        print(f"  原始: {old_correct}/{old_total} = {old_accuracy:.2f}%")
        print(f"  新提取: {new_correct}/{new_total} = {new_accuracy:.2f}%")
        print(f"  提升: {improvement:+d} ({new_accuracy - old_accuracy:+.2f}%)")
        
        # 保存更新后的结果
        output_file = json_file.parent / f"{json_file.stem}_re_extracted.json"
        data['correct_new'] = new_correct
        data['total_new'] = new_total
        data['accuracy_new'] = new_accuracy
        data['improvement'] = improvement
        data['improvement_pct'] = new_accuracy - old_accuracy
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  已保存到: {output_file.name}")
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    old_total_accuracy = (total_old_correct / total_samples * 100) if total_samples > 0 else 0.0
    new_total_accuracy = (total_new_correct / total_samples * 100) if total_samples > 0 else 0.0
    print(f"总样本数: {total_samples}")
    print(f"原始准确率: {total_old_correct}/{total_samples} = {old_total_accuracy:.2f}%")
    print(f"新提取准确率: {total_new_correct}/{total_samples} = {new_total_accuracy:.2f}%")
    print(f"总提升: {total_improvement:+d} ({new_total_accuracy - old_total_accuracy:+.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        round_dir = sys.argv[1]
    else:
        round_dir = "results/round1"
    
    re_extract_results(round_dir)

