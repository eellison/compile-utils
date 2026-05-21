# vllm_Qwen_Qwen3-0.6B

Source model: `Qwen/Qwen3-0.6B`

Workload: vLLM/HuggingFace-style inference capture

Repros: 9

Benchmark all regions:

```bash
python scripts/bench.py models/inference/vllm_Qwen_Qwen3-0.6B
```

Recapture with:

```bash
python scripts/extract_vllm.py "Qwen/Qwen3-0.6B" --inference-only --device 0
```

Extraction output lands under `output/aten_repros/`; validate before replacing
this checked-in directory.
