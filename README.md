```bash
# 环境
pip install -r requirements.txt && pip install vllm

# 单个模型评测（使用4个GPU加速）
python batch_eval.py --model-path /volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-32k-1e --gpu 0,1,2,3

# 批量评测所有模型（单轮，自动跳过已完成的模型）
cd /volume/data/rhjiang/math_evaluator
bash batch_eval_all.sh

# 多轮评测（推荐，运行3轮取平均值，结果存储到 round1/round2/round3）
bash batch_eval_rounds.sh

# 或者后台运行多轮评测
nohup bash batch_eval_rounds.sh > eval_rounds_log.txt 2>&1 &

# 查看运行日志
tail -f eval_rounds_log.txt

# 查看结果
ls -lh results/round*/  # 查看各轮结果
cat results/summary_average.json  # 查看平均值汇总
```

