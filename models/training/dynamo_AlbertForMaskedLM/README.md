# dynamo_AlbertForMaskedLM

Source model: `AlbertForMaskedLM` from the PyTorch benchmark/dynamo
HuggingFace harness

Workload: training-oriented capture

Repros: 32

Benchmark all regions:

```bash
python scripts/bench.py models/training/dynamo_AlbertForMaskedLM
```

Recapture with:

```bash
python scripts/extract_reductions.py dynamo:AlbertForMaskedLM --mode aten
```

This path expects the PyTorch benchmark/dynamo helper modules through
`PYTORCH_DIR` or `/tmp/pytorch-work`.
