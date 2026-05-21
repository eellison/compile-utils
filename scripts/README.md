# Scripts

This directory contains the active extraction and benchmarking utilities.

| Script | Purpose |
| --- | --- |
| `bench.py` | Benchmark one repro or every `*/repro.py` under a directory. |
| `extract_reductions.py` | Core extraction implementation for scheduler and ATen-level region capture. |
| `extract_vllm.py` | vLLM/HuggingFace-oriented model driver built on ATen extraction. |

Removed legacy entry points should not be reintroduced unless they are updated
to the current `models/**/region_*/repro.py` layout.

## `bench.py`

Examples:

```bash
python scripts/bench.py models/inference/vllm_openai_gpt-oss-20b
python scripts/bench.py models/training/dynamo_BertForMaskedLM --no-cuda-graph
python scripts/bench.py models/inference/vllm_Qwen_Qwen3-0.6B/region_002_mean_4bbcddf61f1a_cffba909/repro.py --update-meta
```

The benchmark reports:

- total input plus output bytes
- generated kernel count
- same-size memcopy SOL
- default compiled time
- coord-descent compiled time
- default and coord-descent gaps versus SOL

By default compiled timings use CUDA graph replay. Pass `--no-cuda-graph` to use
`triton.testing.do_bench` for compiled timings.

## `extract_reductions.py`

Use ATen mode for the curated corpus:

```bash
python scripts/extract_reductions.py dynamo:BertForMaskedLM --mode aten
python scripts/extract_reductions.py dynamo:list
```

ATen extraction captures post-grad FX graphs, partitions fusible operators with
Inductor's fusibility rules, emits one `repro.py` per region, and writes an
`index.json` in the generated output directory.

The script also contains older scheduler extraction support. Prefer `--mode
aten` for new model-derived repros unless you are intentionally comparing
against scheduler-level extraction.

## `extract_vllm.py`

Use this driver for vLLM/HuggingFace-style inference models:

```bash
python scripts/extract_vllm.py "openai/gpt-oss-20b" --inference-only --device 0
python scripts/extract_vllm.py --list
```

The script instantiates models from config, optionally reduces layer counts for
large models, creates synthetic inputs, and delegates extraction to
`run_aten_extraction()`.

Generated captures are written under `output/aten_repros/`; validate before
promoting them into `models/`.
