# compile-utils

Standalone repros and benchmarking tools for torch.compile/inductor kernel regions.

## What this is

When you `torch.compile` a model, inductor splits computation into fused kernel
regions. Each region becomes a Triton (or C++) kernel. Some of those kernels are
slower than they should be — the heuristics pick the wrong reduction strategy,
tiling config, or fail to fuse things that should be fused.

This repo contains:
1. **Extracted repros** — standalone `.py` files for every kernel region of a model
2. **Benchmark data** — measured perf vs coord-descent (best achievable with same code shape) and memcopy SOL
3. **Tooling** — scripts to extract, benchmark, and diagnose

## How extraction works

`extract_reductions.py` hooks `inductor_config.post_grad_custom_pre_pass` to capture
the post-decomposition FX graph, then uses `CapabilityBasedPartitioner` with
`is_fusible_node` to partition it into fusion regions — the same groupings inductor
would compile into individual kernels. For each region it:

1. Extracts the partition subgraph as a standalone `torch.nn.Module` with inputs
   matching the original shapes/dtypes/strides
2. Merges reduction partitions that share common inputs (mix-order reduction)
3. Wraps it in a runnable script with a `benchmark()` function that measures:
   - Compiled (default heuristics) time
   - Coord-descent tuned time (best config, same kernel structure)
   - Memcopy SOL at the same transfer size (bandwidth ceiling)
   - Number of Triton kernels generated

Each region gets a content-addressed hash (FX op pattern + input shapes),
so re-extracting the same model produces stable directory names.

`extract_vllm.py` drives this for HuggingFace/vLLM models — instantiates the model
from config (no weights needed), creates dummy inputs, and compiles with the
capture hook active.

## How benchmarking works

Each `repro.py` is self-contained. Running it directly prints perf numbers:

```
$ python models/inference/vllm_openai_gpt-oss-20b/region_011_.../repro.py

Kernel data: 525312.0 KB (read+write)
Compiled (default):      152.3 us
Compiled (coord descent):137.8 us
Memcopy SOL (same size): 167.7 us  (6413.4 GB/s)
Gap (default / SOL):     0.91x
Gap (CD / SOL):          0.82x
```

Key metric: **compiled / coord_descent**. If this ratio is >1.15x, the default
heuristics are leaving perf on the table and the kernel is "actionable."

`benchmark_all.py` runs all repros in a model dir and writes `benchmark_results.json`.

## Structure

```
models/
  inference/           # inference workloads (latency-sensitive)
    vllm_openai_gpt-oss-20b/
      region_011_amax_sum_<hash>/
        repro.py       # standalone, runnable, no repo imports
        meta.json      # hash, ops, num_kernels, perf by hardware
  training/            # training workloads (throughput-sensitive)
    dynamo_AlbertForMaskedLM/
      ...

scripts/
  sol_gap.py           # unified CLI: extract, bench, report, investigate
  extract_reductions.py  # core extraction hook
  extract_vllm.py      # model instantiation for vLLM/HF models
  benchmark_all.py     # batch benchmark runner
  investigate_kernel.py  # dump generated Triton code for a region
```

## meta.json

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

## Usage

```bash
# Extract regions from a model
python scripts/sol_gap.py extract --model "openai/gpt-oss-20b" --mode inference

# Benchmark all regions in a model dir
python scripts/sol_gap.py bench --dir models/inference/vllm_openai_gpt-oss-20b

# Show gaps sorted by severity
python scripts/sol_gap.py report --min-gap 1.3

# Investigate a specific kernel (dumps generated Triton)
python scripts/sol_gap.py investigate models/inference/.../repro.py
```

## Tracking improvements

Kernels with headroom are tracked as GitHub issues on this repo.
Fixes land in pytorch/pytorch.
