```bash
# 环境
pip install -r requirements.txt && pip install vllm

#评测
python batch_eval.py --model-path /volume/data/lzh/Pai-Megatron-Patch-main/qwen-ckpts/Qwen3-4B-base-openr1-default-32k-1e

#结果
ls -lh results/  
```

