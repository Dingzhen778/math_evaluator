# Math Benchmark Evaluation Framework

## 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install vllm math-verify
```

## 评测方法

### 小模型评测 (1.5B-7B 参数量)

对于小模型（如 Qwen2.5-Math-1.5B），**每张GPU运行一个模型实例**是最优策略：
- 单卡推理已经足够快
- 多卡并行可以同时评测多个模型或多轮实验
- 避免不必要的 tensor parallelism 开销

```bash
# 4张GPU并行评测2个模型各2轮（推荐）
source venv/bin/activate

# GPU 0: 模型1
CUDA_VISIBLE_DEVICES=0 python run_eval.py --dataset all \
  --model-path /path/to/model1 \
  --results-dir results > log_gpu0.txt 2>&1 &

# GPU 1: 模型2
CUDA_VISIBLE_DEVICES=1 python run_eval.py --dataset all \
  --model-path /path/to/model2 \
  --results-dir results > log_gpu1.txt 2>&1 &

# GPU 2: 模型1 (第2轮)
CUDA_VISIBLE_DEVICES=2 python run_eval.py --dataset all \
  --model-path /path/to/model1 \
  --results-dir results > log_gpu2.txt 2>&1 &

# GPU 3: 模型2 (第2轮)
CUDA_VISIBLE_DEVICES=3 python run_eval.py --dataset all \
  --model-path /path/to/model2 \
  --results-dir results > log_gpu3.txt 2>&1 &
```

### 大模型评测 (70B+ 参数量)

对于大模型需要使用 tensor parallelism 跨多卡推理：

```bash
# 使用4张GPU进行tensor parallel推理
python run_eval.py --dataset all \
  --model-path /path/to/70b-model \
  --gpu 0,1,2,3 \
  --results-dir results
```

## 支持的数据集

- `math500`: MATH 500题子集
- `gsm8k`: GSM8K 测试集
- `aime2024`: AIME 2024
- `aime2025`: AIME 2025
- `gaokao2023en`: 高考2023英文版
- `mathodyssey`: MathOdyssey
- `olympiadbench_oe`: OlympiadBench 开放式题目
- `all`: 评测所有数据集 (math500, gsm8k, aime2024, aime2025)

## 查看结果

```bash
# 查看评测进度
tail -f log_gpu0.txt

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看结果文件
ls -lh results/
```

## 结果格式

每个评测结果保存为JSON文件：
```json
{
  "dataset": "math500",
  "accuracy": 66.60,
  "correct": 333,
  "total": 500,
  "results": [...]
}
```
