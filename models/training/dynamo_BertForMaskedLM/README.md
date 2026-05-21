# dynamo_BertForMaskedLM

Source model: `BertForMaskedLM` from the PyTorch benchmark/dynamo HuggingFace
harness

Workload: training-oriented capture

Repros: 26

Benchmark all regions:

```bash
python scripts/bench.py models/training/dynamo_BertForMaskedLM
```

Recapture with:

```bash
python scripts/extract_reductions.py dynamo:BertForMaskedLM --mode aten
```

This path expects the PyTorch benchmark/dynamo helper modules through
`PYTORCH_DIR` or `/tmp/pytorch-work`.
