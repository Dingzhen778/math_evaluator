# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Math benchmark evaluation framework for evaluating LLM mathematical reasoning capabilities. Supports both local vLLM inference and remote OpenAI-compatible APIs.

## Common Commands

### Setup
```bash
bash install_deps.sh
# Or manually:
pip install tqdm requests math-verify datasets transformers
pip install vllm==0.11.0 --no-deps  # avoid torch version conflicts
```

### Running Evaluations

Local model (vLLM):
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python run_eval.py --dataset math500 \
  --model-path /path/to/model --results-dir results

# Multi-GPU tensor parallel (for 70B+ models)
python run_eval.py --dataset all --model-path /path/to/70b-model --gpu 0,1,2,3

# All 8 datasets
python run_eval.py --dataset all --model-path /path/to/model
```

Remote API:
```bash
export EVAL_BASE_URL=https://api.example.com/v1
export EVAL_MODEL=model-name
export EVAL_API_KEY=your-key
python run_eval.py --dataset gsm8k
```

### Datasets
`math500`, `gsm8k`, `aime2024`, `aime2025`, `gaokao2023en`, `mathodyssey`, `amc23`, `olympiadbench_oe`, `all`

## Architecture

### Core Components

**Entry Point**: `run_eval.py`
- Parses CLI args, initializes client (VLLMClient or APIClient), runs evaluation
- Loads model once and reuses across datasets when `--dataset all`

**Evaluator**: `evaluators/evaluator.py` - `MathEvaluator`
- Main orchestrator with `DATASET_CONFIGS` dict mapping dataset names to components
- Supports both single-threaded and batch evaluation (`_evaluate_batch`)
- Uses ThreadPoolExecutor for API-based evaluation, batch processing for vLLM

**Base Classes**: `evaluators/base.py`
- `EvalSample`, `EvalResult` - data classes
- `BaseDatasetLoader`, `BaseAnswerExtractor`, `BaseAnswerValidator`, `BasePromptBuilder` - abstract interfaces

### Data Flow

1. **Dataset Loading** (`evaluators/datasets/`): Loaders return `List[EvalSample]` from HuggingFace datasets
2. **Prompt Building** (`evaluators/prompts/templates.py`): `MATHPromptBuilder`, `GSM8KPromptBuilder`, `AImePromptBuilder` build system/user prompts with few-shot examples
3. **Inference**: `VLLMClient.generate_batch()` or `APIClient.generate()`
4. **Answer Extraction** (`evaluators/extractors/`): Extract final answer from model output (e.g., `\boxed{}` or `####`)
5. **Validation** (`evaluators/validators/math_verify_validator.py`): Uses `math-verify` library for symbolic math verification with fallback string comparison

### Clients

- `VLLMClient` (`evaluators/vllm_client.py`): Direct vLLM Python API, thread-safe with locks, supports `generate_batch()` for high throughput
- `APIClient` (`evaluators/api_client.py`): OpenAI-compatible HTTP client with retry logic

### Adding a New Dataset

1. Create loader in `evaluators/datasets/` extending `BaseDatasetLoader`
2. Add to `evaluators/datasets/__init__.py`
3. Add config entry to `MathEvaluator.DATASET_CONFIGS` in `evaluators/evaluator.py`
4. Update dataset choices in `run_eval.py` argparse

### Key Files
- `evaluators/validators/math_verify_validator.py`: Answer verification using `math-verify` with LaTeX/expression extraction
- `evaluators/prompts/templates.py`: Few-shot prompt templates for each dataset type
