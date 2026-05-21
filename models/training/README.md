# Training Captures

This directory contains training-oriented fusion-region repros from the PyTorch
benchmark/dynamo HuggingFace harness. Captures can include both forward regions
and backward regions, depending on what the compiler captures for the workload.

| Model directory | Source model | Repros |
| --- | --- | ---: |
| `dynamo_AlbertForMaskedLM` | `AlbertForMaskedLM` | 32 |
| `dynamo_BertForMaskedLM` | `BertForMaskedLM` | 26 |

## Benchmarking

Benchmark one model directory:

```bash
python scripts/bench.py models/training/dynamo_BertForMaskedLM
```

Benchmark one region and update its metadata:

```bash
python scripts/bench.py \
  models/training/dynamo_AlbertForMaskedLM/region_000_amax_sum_eb4fe3ac03e0_d46ab65d/repro.py \
  --update-meta
```

## Recapturing

Use ATen extraction mode for model-derived training captures:

```bash
python scripts/extract_reductions.py dynamo:BertForMaskedLM --mode aten
python scripts/extract_reductions.py dynamo:AlbertForMaskedLM --mode aten
python scripts/extract_reductions.py dynamo:list
```

`dynamo:*` extraction expects access to the PyTorch benchmark/dynamo helper
modules through `PYTORCH_DIR` or the default `/tmp/pytorch-work`.
