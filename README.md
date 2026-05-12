# compile-utils

Standalone repros and benchmarking tools for torch.compile/inductor kernel regions.

## Structure

```
models/
  inference/           # inference workloads (latency-sensitive)
    vllm_openai_gpt-oss-20b/
      region_011_amax_sum_.../
        repro.py       # standalone, runnable
        meta.json      # hash, ops, perf, num_kernels
  training/            # training workloads (throughput-sensitive)
    dynamo_AlbertForMaskedLM/
      ...

scripts/
  sol_gap.py           # unified CLI: extract, bench, report, investigate
  extract_reductions.py
  extract_vllm.py
  benchmark_all.py
  investigate_kernel.py
```

## Usage

```bash
# Extract regions from a model
python scripts/sol_gap.py extract --model "openai/gpt-oss-20b" --mode inference

# Benchmark all regions
python scripts/sol_gap.py bench --dir models/inference/vllm_openai_gpt-oss-20b

# Show gaps sorted by severity
python scripts/sol_gap.py report --min-gap 1.3

# Investigate a specific kernel
python scripts/sol_gap.py investigate models/inference/.../repro.py
```

## Tracking improvements

Actionable kernels (where coord-descent significantly outperforms default heuristics)
are tracked as GitHub issues on this repo. Each issue links to the repro and describes
the suspected heuristic problem. Fixes land in pytorch/pytorch.
