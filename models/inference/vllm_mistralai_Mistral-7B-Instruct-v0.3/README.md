# vllm_mistralai_Mistral-7B-Instruct-v0.3

Source model: `mistralai/Mistral-7B-Instruct-v0.3`

Workload: vLLM/HuggingFace-style inference capture

Repros: 9

Benchmark all regions:

```bash
python scripts/bench.py models/inference/vllm_mistralai_Mistral-7B-Instruct-v0.3
```

Recapture with:

```bash
python scripts/extract_vllm.py "mistralai/Mistral-7B-Instruct-v0.3" --inference-only --device 0
```

`extract_vllm.py` may reduce layer count for this model to keep capture
practical on one GPU.
