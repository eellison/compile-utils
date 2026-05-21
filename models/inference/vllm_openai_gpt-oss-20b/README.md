# vllm_openai_gpt-oss-20b

Source model: `openai/gpt-oss-20b`

Workload: vLLM/HuggingFace-style inference capture

Repros: 15

Benchmark all regions:

```bash
python scripts/bench.py models/inference/vllm_openai_gpt-oss-20b
```

Recapture with:

```bash
python scripts/extract_vllm.py "openai/gpt-oss-20b" --inference-only --device 0
```

`extract_vllm.py` layer-caps this model by default so it can fit on one GPU.
