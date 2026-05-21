# Inference Captures

This directory contains forward/inference fusion-region repros from
vLLM-oriented HuggingFace model configs. The captures use synthetic inputs and
do not require model weights.

| Model directory | Source model | Repros |
| --- | --- | ---: |
| `vllm_Qwen_Qwen3-0.6B` | `Qwen/Qwen3-0.6B` | 9 |
| `vllm_deepseek-ai_DeepSeek-V3` | `deepseek-ai/DeepSeek-V3` | 28 |
| `vllm_mistralai_Mistral-7B-Instruct-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | 9 |
| `vllm_openai_gpt-oss-20b` | `openai/gpt-oss-20b` | 15 |

Large models may be captured with fewer layers to fit on one GPU. The extracted
region shapes and strides are still taken from the actual traced graph.

## Benchmarking

Benchmark one model directory:

```bash
python scripts/bench.py models/inference/vllm_openai_gpt-oss-20b
```

Benchmark one region and update its metadata:

```bash
python scripts/bench.py \
  models/inference/vllm_Qwen_Qwen3-0.6B/region_002_mean_4bbcddf61f1a_cffba909/repro.py \
  --update-meta
```

## Recapturing

Use `scripts/extract_vllm.py` for this family:

```bash
python scripts/extract_vllm.py "Qwen/Qwen3-0.6B" --inference-only --device 0
python scripts/extract_vllm.py --list
```

Extraction writes to `output/aten_repros/`. Inspect and validate the generated
regions before replacing a checked-in model directory.
