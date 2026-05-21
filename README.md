# compile-utils

Standalone repro corpus and benchmarking utilities for `torch.compile` /
TorchInductor fusion regions.

This repository is meant to make kernel issues easy to reproduce without
re-running full models. Each checked-in repro is a small `torch.nn.Module`
extracted from a real model graph, with synthetic inputs matching the captured
shape, dtype, and stride metadata.

## What Is Checked In

The current corpus has 119 repros from model captures:

| Area | Model directory | Repros |
| --- | --- | ---: |
| Inference | `models/inference/vllm_Qwen_Qwen3-0.6B` | 9 |
| Inference | `models/inference/vllm_deepseek-ai_DeepSeek-V3` | 28 |
| Inference | `models/inference/vllm_mistralai_Mistral-7B-Instruct-v0.3` | 9 |
| Inference | `models/inference/vllm_openai_gpt-oss-20b` | 15 |
| Training | `models/training/dynamo_AlbertForMaskedLM` | 32 |
| Training | `models/training/dynamo_BertForMaskedLM` | 26 |

Each model directory contains one directory per fusion region:

```text
models/inference/vllm_openai_gpt-oss-20b/
  region_011_amax_sum_c0c1b95fe65e_63302950/
    repro.py
    meta.json
```

`repro.py` defines:

- `Repro`: the extracted module
- `make_inputs()`: synthetic CUDA inputs for the captured signature

`meta.json` records the stable hash, operator list, reduction kinds, current
kernel count, and optional benchmark results by hardware.

## Why This Exists

When Inductor compiles a model, it decomposes the model graph, partitions
fusible ATen operators, and lowers each partition to generated kernels. A full
model is often too large and noisy for compiler iteration. These repros let us:

- reproduce a single partition without model weights
- measure default Inductor codegen against coord-descent tuned codegen
- compare kernel time to same-size memcopy SOL
- inspect missed fusion, split regions, reduction strategy, and tiling issues
- keep a stable corpus while compiler changes evolve

## Quickstart

Benchmark one repro:

```bash
python scripts/bench.py \
  models/inference/vllm_openai_gpt-oss-20b/region_011_amax_sum_c0c1b95fe65e_63302950/repro.py
```

Benchmark every repro under a model directory:

```bash
python scripts/bench.py models/inference/vllm_openai_gpt-oss-20b
```

Update a region's `meta.json` with fresh B200 numbers:

```bash
python scripts/bench.py \
  models/inference/vllm_openai_gpt-oss-20b/region_011_amax_sum_c0c1b95fe65e_63302950/repro.py \
  --update-meta
```

Use do_bench instead of CUDA graph replay:

```bash
python scripts/bench.py models/training/dynamo_BertForMaskedLM --no-cuda-graph
```

## Requirements

The checked-in repros are self-contained apart from runtime dependencies:

- a CUDA-capable PyTorch build
- Triton
- a GPU visible to PyTorch

Extraction scripts also need the model-side dependencies they instantiate
(`transformers`, vLLM-relevant configs, and optionally a PyTorch checkout for
`benchmarks/dynamo` helpers). `PYTORCH_DIR` defaults to `/tmp/pytorch-work` when
a PyTorch checkout is needed.

## Extraction

The active extraction path is `scripts/extract_reductions.py`. Its ATen mode
captures post-grad FX graphs through `inductor_config.post_grad_custom_pre_pass`,
then partitions with Inductor's `is_fusible_node` rules into kernel-sized
regions. Training captures include forward regions and, when available,
backward regions.

For vLLM/HuggingFace-style inference captures:

```bash
python scripts/extract_vllm.py "openai/gpt-oss-20b" --inference-only --device 0
```

List the configured vLLM-oriented model set:

```bash
python scripts/extract_vllm.py --list
```

For PyTorch benchmark/dynamo HuggingFace captures:

```bash
python scripts/extract_reductions.py dynamo:BertForMaskedLM --mode aten
python scripts/extract_reductions.py dynamo:list
```

New extraction output is written under generated `output/` directories. Inspect
new captures before promoting them into `models/inference/` or
`models/training/`.

## Benchmarking

`scripts/bench.py` is the supported benchmark entry point. For each repro it:

1. Loads `Repro` and `make_inputs()`.
2. Runs eager once to count input and output bytes.
3. Compiles the module and counts generated kernel launches.
4. Measures same-size memcopy SOL with `triton.testing.do_bench`.
5. Measures default Inductor compiled time.
6. Measures coord-descent tuned compiled time.

By default compiled timings use CUDA graph replay to reduce Python dispatch
noise. `--no-cuda-graph` switches compiled timing back to `do_bench`.

Example output:

```text
Kernel data: 12345.0 KB (read+write)
Kernels generated: 2
Memcopy SOL (same size):      8.1 us
Compiled (default):          12.4 us
Compiled (coord desc):       10.9 us
Gap (default / SOL):          1.53x
Gap (CD / SOL):               1.35x
```

## Metadata

`meta.json` is intentionally small and stable:

```json
{
  "hash": "c0c1b95fe65e_63302950",
  "ops": ["aten.amax.default", "aten.sum.dim_IntList"],
  "reduction_types": ["amax", "sum"],
  "num_kernels": 6,
  "perf": {
    "B200": {
      "compiled_us": 152.3,
      "coord_descent_us": 137.8,
      "memcpy_sol_us": 167.7,
      "total_bytes": 537919488
    }
  }
}
```

Keep benchmark values hardware-scoped. Do not overwrite one GPU generation's
numbers with another GPU generation's numbers under the same key.

## Directory Guides

- `models/README.md`: corpus layout and promotion rules
- `models/inference/README.md`: inference model captures
- `models/training/README.md`: training model captures
- `scripts/README.md`: supported scripts and workflows

## Generated Files

Generated extraction output, benchmark summaries, `__pycache__`, and
investigation logs should stay out of git. The checked-in corpus should contain
only curated model directories, region `repro.py` files, `meta.json`, and docs.

## Tracking Improvements

Use GitHub issues for kernels with clear compiler headroom or correctness
failures. Fixes usually land in `pytorch/pytorch`; this repo keeps the small
repros and measurement history needed to validate those fixes.
