# vllm_deepseek-ai_DeepSeek-V3

Source model: `deepseek-ai/DeepSeek-V3`

Workload: vLLM/HuggingFace-style inference capture

Repros: 28

Benchmark all regions:

```bash
python scripts/bench.py models/inference/vllm_deepseek-ai_DeepSeek-V3
```

Recapture with:

```bash
python scripts/extract_vllm.py "deepseek-ai/DeepSeek-V3" --inference-only --device 0
```

`extract_vllm.py` layer-caps this model by default so it can fit on one GPU
while preserving representative kernel patterns.
