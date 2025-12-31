# Math Evaluator Results Analysis

生成时间: 2025-12-27 00:59:24

## 总体统计

- **总评测记录数**: 1205
- **不同模型数**: 64
- **数据集数**: 8
- **结果目录数**: 8

### 按模型类型统计

| 类型 | 模型数 |
|------|--------|
| LLaMA | 18 |
| Qwen2.5 | 5 |
| Qwen2.5-Math | 20 |
| Qwen3 | 21 |

### 按目录统计

| 目录 | 评测文件数 | 模型数 |
|------|------------|--------|
| results | 241 | 12 |
| results-1222 | 120 | 6 |
| results-1222-mathverify | 62 | 8 |
| results-1222-new | 120 | 6 |
| results-1223 | 169 | 20 |
| results-1224 | 357 | 15 |
| results-1225 | 56 | 7 |
| results-1227 | 80 | 10 |

---

## 详细模型列表


### LLaMA 模型 (18个)


#### llama3_1-8B-MetaMathQA-all

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 7.40% | 12.40% | 13.80% |
| gsm8k | 72.63% | 76.72% | 77.03% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 23.12% | 22.60% | 27.01% |
| mathodyssey | 7.71% | 8.23% | 7.97% |
| amc23 | 7.50% | 10.00% | 7.50% |
| olympiadbench_oe | 6.52% | 4.89% | 6.22% |

#### llama3_1-8B-metamathqa-method-reliable

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 14.20% | 16.00% | 15.80% |
| gsm8k | 73.69% | 73.77% | 75.89% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 21.56% | 24.16% | 23.64% |
| mathodyssey | 7.46% | 7.97% | 7.46% |
| amc23 | 10.00% | 7.50% | 15.00% |
| olympiadbench_oe | 5.78% | 5.93% | 6.81% |

#### llama3_1-8B-metamathqa-nogreedy-reliable

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 28.00% | 30.80% | 32.60% |
| gsm8k | 69.83% | 74.22% | 75.13% |
| aime2024 | 0.00% | 0.00% | 3.33% |
| aime2025 | 3.33% | 0.00% | 0.00% |
| gaokao2023en | 23.38% | 24.16% | 22.60% |
| mathodyssey | 7.97% | 6.68% | 6.43% |
| amc23 | 5.00% | 5.00% | 10.00% |
| olympiadbench_oe | 5.33% | 6.67% | 6.96% |

#### llama3_1-8B-metamathqa-random-reliable

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 27.40% | 30.40% | 31.40% |
| gsm8k | 71.34% | 74.83% | 74.83% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 21.82% | 23.38% | 23.90% |
| mathodyssey | 6.94% | 5.91% | 6.68% |
| amc23 | 5.00% | 10.00% | 5.00% |
| olympiadbench_oe | 5.63% | 5.63% | 5.93% |

#### llama3_1-8B-numinamath-all

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 40.80% | 37.20% | 31.20% |
| gsm8k | 64.82% | 72.10% | 73.92% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 3.33% | 0.00% | 0.00% |
| gaokao2023en | 41.04% | 32.73% | 31.43% |
| mathodyssey | 17.22% | 20.82% | 17.74% |
| amc23 | 17.50% | 20.00% | 20.00% |
| olympiadbench_oe | 11.56% | 15.26% | 17.04% |

#### llama3_1-8B-numinamath-method-reliable

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 39.60% | 41.20% | 44.80% |
| gsm8k | 61.33% | 71.34% | 64.59% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 3.33% | 0.00% |
| gaokao2023en | 35.32% | 40.00% | 36.10% |
| mathodyssey | 19.79% | 19.54% | 21.08% |
| amc23 | 17.50% | 20.00% | 15.00% |
| olympiadbench_oe | 8.89% | 12.30% | 11.26% |

### Qwen2.5-Math 模型 (20个)


#### Qwen2.5-Math-1.5B

- **Epochs**: N/A
- **数据集覆盖**: 7/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 66.60% |
| gsm8k | 78.85% |
| aime2024 | 10.00% |
| aime2025 | 6.67% |
| gaokao2023en | 58.70% |
| mathodyssey | 25.71% |
| amc23 | - |
| olympiadbench_oe | 30.81% |

#### Qwen2.5-Math-1.5B-Instruct

- **Epochs**: N/A
- **数据集覆盖**: 7/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 75.20% |
| gsm8k | 78.09% |
| aime2024 | 13.33% |
| aime2025 | 6.67% |
| gaokao2023en | 65.71% |
| mathodyssey | 41.90% |
| amc23 | - |
| olympiadbench_oe | 39.11% |

#### Qwen2.5-math-1.5B-metamathqa-all-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 39.60% | 33.60% | 39.00% |
| gsm8k | 52.92% | 48.67% | 42.00% |
| aime2024 | 0.00% | 0.00% | 6.67% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 35.06% | 33.25% | 37.66% |
| mathodyssey | 17.22% | 17.48% | 15.68% |
| amc23 | 20.00% | 7.50% | 10.00% |
| olympiadbench_oe | 12.15% | 10.37% | 14.37% |

#### Qwen2.5-math-1.5B-metamathqa-all-unpack-32k

- **Epochs**: 1e
- **数据集覆盖**: 7/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 36.00% |
| gsm8k | 64.59% |
| aime2024 | 0.00% |
| aime2025 | 0.00% |
| gaokao2023en | 31.43% |
| mathodyssey | 13.11% |
| amc23 | - |
| olympiadbench_oe | 11.41% |

#### Qwen2.5-math-1.5B-metamathqa-reliable-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 43.40% | 34.60% | 33.40% |
| gsm8k | 49.89% | 50.11% | 46.55% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 36.36% | 35.06% | 34.55% |
| mathodyssey | 17.48% | 17.22% | 16.20% |
| amc23 | 7.50% | 10.00% | 10.00% |
| olympiadbench_oe | 15.11% | 10.96% | 11.11% |

#### Qwen2.5-math-1.5B-metamathqa-reliable-nogreedy-32k

- **Epochs**: 1e
- **数据集覆盖**: 8/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 43.80% |
| gsm8k | 47.38% |
| aime2024 | 0.00% |
| aime2025 | 0.00% |
| gaokao2023en | 37.40% |
| mathodyssey | 16.71% |
| amc23 | 7.50% |
| olympiadbench_oe | 14.81% |

#### Qwen2.5-math-1.5B-metamathqa-reliable-random-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 43.80% | 33.80% | 33.20% |
| gsm8k | 43.75% | 45.26% | 40.41% |
| aime2024 | 3.33% | 0.00% | 0.00% |
| aime2025 | 0.00% | 3.33% | 0.00% |
| gaokao2023en | 37.40% | 35.84% | 33.25% |
| mathodyssey | 16.71% | 16.45% | 17.74% |
| amc23 | 12.50% | 17.50% | 12.50% |
| olympiadbench_oe | 16.89% | 11.85% | 11.70% |

#### Qwen2.5-math-1.5B-numinamath-default-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 55.40% | 50.80% | 48.20% |
| gsm8k | 70.89% | 70.58% | 69.90% |
| aime2024 | 3.33% | 3.33% | 3.33% |
| aime2025 | 0.00% | 3.33% | 0.00% |
| gaokao2023en | 46.23% | 41.82% | 41.56% |
| mathodyssey | 21.34% | 22.37% | 21.34% |
| amc23 | 32.50% | 30.00% | 15.00% |
| olympiadbench_oe | 19.85% | 17.63% | 16.89% |

#### Qwen2.5-math-1.5B-numinamath-reliable-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 59.60% | 54.60% | 54.40% |
| gsm8k | 73.77% | 71.95% | 72.63% |
| aime2024 | 6.67% | 3.33% | 3.33% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 47.01% | 48.57% | 48.31% |
| mathodyssey | 23.14% | 22.62% | 21.59% |
| amc23 | 45.00% | 32.50% | 32.50% |
| olympiadbench_oe | 23.56% | 20.89% | 21.04% |

#### Qwen2.5-math-1.5B-numinamath-reliable-random-32k

- **Epochs**: 1e
- **数据集覆盖**: 8/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 44.20% |
| gsm8k | 45.56% |
| aime2024 | 3.33% |
| aime2025 | 0.00% |
| gaokao2023en | 36.88% |
| mathodyssey | 15.94% |
| amc23 | 12.50% |
| olympiadbench_oe | 16.74% |

### Qwen2.5 模型 (5个)


#### Qwen2.5-1.5B

- **Epochs**: N/A
- **数据集覆盖**: 1/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | - |
| gsm8k | 6.97% |
| aime2024 | - |
| aime2025 | - |
| gaokao2023en | - |
| mathodyssey | - |
| amc23 | - |
| olympiadbench_oe | - |

#### Qwen2.5-1.5B-metamathqa-all-pack-32k

- **Epochs**: 1e
- **数据集覆盖**: 8/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 16.40% |
| gsm8k | 31.69% |
| aime2024 | 0.00% |
| aime2025 | 0.00% |
| gaokao2023en | 16.36% |
| mathodyssey | 5.66% |
| amc23 | 5.00% |
| olympiadbench_oe | 3.85% |

#### Qwen2.5-1.5B-metamathqa-reliable-pack-32k

- **Epochs**: 1e
- **数据集覆盖**: 8/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 14.80% |
| gsm8k | 24.64% |
| aime2024 | 0.00% |
| aime2025 | 0.00% |
| gaokao2023en | 14.55% |
| mathodyssey | 5.14% |
| amc23 | 5.00% |
| olympiadbench_oe | 2.81% |

#### Qwen2.5-1.5B-numinamath-all-32k

- **Epochs**: 1e
- **数据集覆盖**: 8/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 17.40% |
| gsm8k | 46.25% |
| aime2024 | 0.00% |
| aime2025 | 0.00% |
| gaokao2023en | 15.58% |
| mathodyssey | 6.94% |
| amc23 | 2.50% |
| olympiadbench_oe | 3.26% |

#### Qwen2.5-1.5B-numinamath-reliable-32k

- **Epochs**: 1e
- **数据集覆盖**: 8/8

| 数据集 | 准确率 |
|--------|--------|
| math500 | 13.60% |
| gsm8k | 47.61% |
| aime2024 | 0.00% |
| aime2025 | 0.00% |
| gaokao2023en | 11.69% |
| mathodyssey | 4.11% |
| amc23 | 2.50% |
| olympiadbench_oe | 1.19% |

### Qwen3 模型 (21个)


#### Qwen3-4B-base-openr1-default-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 4/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 74.20% | 75.20% | 74.60% |
| gsm8k | 89.92% | 89.46% | 90.30% |
| aime2024 | 10.00% | 10.00% | 10.00% |
| aime2025 | 16.67% | 13.33% | 16.67% |
| gaokao2023en | - | - | - |
| mathodyssey | - | - | - |
| amc23 | - | - | - |
| olympiadbench_oe | - | - | - |

#### Qwen3-4B-base-openr1-default-reliable-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 4/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 75.20% | 75.80% | 73.80% |
| gsm8k | 88.63% | 89.39% | 88.40% |
| aime2024 | 13.33% | 13.33% | 6.67% |
| aime2025 | 6.67% | 16.67% | 10.00% |
| gaokao2023en | - | - | - |
| mathodyssey | - | - | - |
| amc23 | - | - | - |
| olympiadbench_oe | - | - | - |

#### Qwen3-4B-metamathqa-all-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 31.40% | 25.80% | 24.60% |
| gsm8k | 74.00% | 72.71% | 75.44% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 24.94% | 24.42% | 24.42% |
| mathodyssey | 10.80% | 11.57% | 11.57% |
| amc23 | 7.50% | 17.50% | 7.50% |
| olympiadbench_oe | 8.44% | 7.56% | 6.52% |

#### Qwen3-4B-metamathqa-reliable-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 38.00% | 24.40% | 23.40% |
| gsm8k | 60.42% | 61.03% | 66.11% |
| aime2024 | 0.00% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 26.49% | 20.00% | 22.86% |
| mathodyssey | 12.85% | 9.51% | 9.51% |
| amc23 | 22.50% | 20.00% | 15.00% |
| olympiadbench_oe | 11.70% | 9.48% | 8.30% |

#### Qwen3-4B-numinamath-all-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 45.20% | 39.00% | 42.00% |
| gsm8k | 84.91% | 84.84% | 85.29% |
| aime2024 | 6.67% | 3.33% | 6.67% |
| aime2025 | 3.33% | 10.00% | 0.00% |
| gaokao2023en | 32.47% | 30.39% | 27.79% |
| mathodyssey | 11.05% | 11.31% | 12.60% |
| amc23 | 5.00% | 17.50% | 12.50% |
| olympiadbench_oe | 21.78% | 19.26% | 21.04% |

#### Qwen3-4B-numinamath-reliable-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 8/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 22.60% | 22.00% | 27.60% |
| gsm8k | 84.84% | 85.22% | 85.06% |
| aime2024 | 3.33% | 0.00% | 0.00% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 21.82% | 21.04% | 21.04% |
| mathodyssey | 12.34% | 10.28% | 14.91% |
| amc23 | 7.50% | 10.00% | 5.00% |
| olympiadbench_oe | 10.96% | 10.96% | 11.70% |

#### Qwen3-4B-numinamath-reliable-random-32k

- **Epochs**: 1e, 2e, 3e
- **数据集覆盖**: 7/8

| 数据集 | 1e | 2e | 3e |
|--------|-------|-------|-------|
| math500 | 38.80% | 22.60% | 20.00% |
| gsm8k | 62.09% | 56.18% | 68.54% |
| aime2024 | 0.00% | 0.00% | 3.33% |
| aime2025 | 0.00% | 0.00% | 0.00% |
| gaokao2023en | 26.49% | 18.96% | 19.22% |
| mathodyssey | 14.91% | 11.57% | 9.25% |
| amc23 | 22.50% | 32.50% | 10.00% |
| olympiadbench_oe | - | 8.00% | 7.56% |

---

## 完整模型清单

| # | 模型名 | 类型 | 数据集数 | Epoch |
|---|--------|------|----------|-------|
| 1 | Qwen2.5-1.5B | Qwen2.5 | 1/8 | N/A |
| 2 | Qwen2.5-1.5B-metamathqa-all-pack-32k-1e | Qwen2.5 | 8/8 | 1e |
| 3 | Qwen2.5-1.5B-metamathqa-reliable-pack-32k-1e | Qwen2.5 | 8/8 | 1e |
| 4 | Qwen2.5-1.5B-numinamath-all-32k-1e | Qwen2.5 | 8/8 | 1e |
| 5 | Qwen2.5-1.5B-numinamath-reliable-32k-1e | Qwen2.5 | 8/8 | 1e |
| 6 | Qwen2.5-Math-1.5B | Qwen2.5-Math | 7/8 | N/A |
| 7 | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Math | 7/8 | N/A |
| 8 | Qwen2.5-math-1.5B-metamathqa-all-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 9 | Qwen2.5-math-1.5B-metamathqa-all-32k-2e | Qwen2.5-Math | 8/8 | 2e |
| 10 | Qwen2.5-math-1.5B-metamathqa-all-32k-3e | Qwen2.5-Math | 8/8 | 3e |
| 11 | Qwen2.5-math-1.5B-metamathqa-all-unpack-32k-1e | Qwen2.5-Math | 7/8 | 1e |
| 12 | Qwen2.5-math-1.5B-metamathqa-reliable-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 13 | Qwen2.5-math-1.5B-metamathqa-reliable-32k-2e | Qwen2.5-Math | 8/8 | 2e |
| 14 | Qwen2.5-math-1.5B-metamathqa-reliable-32k-3e | Qwen2.5-Math | 8/8 | 3e |
| 15 | Qwen2.5-math-1.5B-metamathqa-reliable-nogreedy-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 16 | Qwen2.5-math-1.5B-metamathqa-reliable-random-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 17 | Qwen2.5-math-1.5B-metamathqa-reliable-random-32k-2e | Qwen2.5-Math | 8/8 | 2e |
| 18 | Qwen2.5-math-1.5B-metamathqa-reliable-random-32k-3e | Qwen2.5-Math | 8/8 | 3e |
| 19 | Qwen2.5-math-1.5B-numinamath-default-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 20 | Qwen2.5-math-1.5B-numinamath-default-32k-2e | Qwen2.5-Math | 8/8 | 2e |
| 21 | Qwen2.5-math-1.5B-numinamath-default-32k-3e | Qwen2.5-Math | 8/8 | 3e |
| 22 | Qwen2.5-math-1.5B-numinamath-reliable-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 23 | Qwen2.5-math-1.5B-numinamath-reliable-32k-2e | Qwen2.5-Math | 8/8 | 2e |
| 24 | Qwen2.5-math-1.5B-numinamath-reliable-32k-3e | Qwen2.5-Math | 8/8 | 3e |
| 25 | Qwen2.5-math-1.5B-numinamath-reliable-random-32k-1e | Qwen2.5-Math | 8/8 | 1e |
| 26 | Qwen3-4B-base-openr1-default-32k-1e | Qwen3 | 4/8 | 1e |
| 27 | Qwen3-4B-base-openr1-default-32k-2e | Qwen3 | 4/8 | 2e |
| 28 | Qwen3-4B-base-openr1-default-32k-3e | Qwen3 | 4/8 | 3e |
| 29 | Qwen3-4B-base-openr1-default-reliable-32k-1e | Qwen3 | 4/8 | 1e |
| 30 | Qwen3-4B-base-openr1-default-reliable-32k-2e | Qwen3 | 4/8 | 2e |
| 31 | Qwen3-4B-base-openr1-default-reliable-32k-3e | Qwen3 | 4/8 | 3e |
| 32 | Qwen3-4B-metamathqa-all-32k-1e | Qwen3 | 8/8 | 1e |
| 33 | Qwen3-4B-metamathqa-all-32k-2e | Qwen3 | 8/8 | 2e |
| 34 | Qwen3-4B-metamathqa-all-32k-3e | Qwen3 | 8/8 | 3e |
| 35 | Qwen3-4B-metamathqa-reliable-32k-1e | Qwen3 | 8/8 | 1e |
| 36 | Qwen3-4B-metamathqa-reliable-32k-2e | Qwen3 | 8/8 | 2e |
| 37 | Qwen3-4B-metamathqa-reliable-32k-3e | Qwen3 | 8/8 | 3e |
| 38 | Qwen3-4B-numinamath-all-32k-1e | Qwen3 | 8/8 | 1e |
| 39 | Qwen3-4B-numinamath-all-32k-2e | Qwen3 | 8/8 | 2e |
| 40 | Qwen3-4B-numinamath-all-32k-3e | Qwen3 | 8/8 | 3e |
| 41 | Qwen3-4B-numinamath-reliable-32k-1e | Qwen3 | 8/8 | 1e |
| 42 | Qwen3-4B-numinamath-reliable-32k-2e | Qwen3 | 8/8 | 2e |
| 43 | Qwen3-4B-numinamath-reliable-32k-3e | Qwen3 | 8/8 | 3e |
| 44 | Qwen3-4B-numinamath-reliable-random-32k-1e | Qwen3 | 7/8 | 1e |
| 45 | Qwen3-4B-numinamath-reliable-random-32k-2e | Qwen3 | 8/8 | 2e |
| 46 | Qwen3-4B-numinamath-reliable-random-32k-3e | Qwen3 | 8/8 | 3e |
| 47 | llama3_1-8B-MetaMathQA-all-1e | LLaMA | 8/8 | 1e |
| 48 | llama3_1-8B-MetaMathQA-all-2e | LLaMA | 8/8 | 2e |
| 49 | llama3_1-8B-MetaMathQA-all-3e | LLaMA | 8/8 | 3e |
| 50 | llama3_1-8B-metamathqa-method-reliable-1e | LLaMA | 8/8 | 1e |
| 51 | llama3_1-8B-metamathqa-method-reliable-2e | LLaMA | 8/8 | 2e |
| 52 | llama3_1-8B-metamathqa-method-reliable-3e | LLaMA | 8/8 | 3e |
| 53 | llama3_1-8B-metamathqa-nogreedy-reliable-1e | LLaMA | 8/8 | 1e |
| 54 | llama3_1-8B-metamathqa-nogreedy-reliable-2e | LLaMA | 8/8 | 2e |
| 55 | llama3_1-8B-metamathqa-nogreedy-reliable-3e | LLaMA | 8/8 | 3e |
| 56 | llama3_1-8B-metamathqa-random-reliable-1e | LLaMA | 8/8 | 1e |
| 57 | llama3_1-8B-metamathqa-random-reliable-2e | LLaMA | 8/8 | 2e |
| 58 | llama3_1-8B-metamathqa-random-reliable-3e | LLaMA | 8/8 | 3e |
| 59 | llama3_1-8B-numinamath-all-1e | LLaMA | 8/8 | 1e |
| 60 | llama3_1-8B-numinamath-all-2e | LLaMA | 8/8 | 2e |
| 61 | llama3_1-8B-numinamath-all-3e | LLaMA | 8/8 | 3e |
| 62 | llama3_1-8B-numinamath-method-reliable-1e | LLaMA | 8/8 | 1e |
| 63 | llama3_1-8B-numinamath-method-reliable-2e | LLaMA | 8/8 | 2e |
| 64 | llama3_1-8B-numinamath-method-reliable-3e | LLaMA | 8/8 | 3e |