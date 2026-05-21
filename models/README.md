# Repro Corpus

`models/` contains the curated, checked-in repro corpus.

The tree is split by workload intent:

- `inference/`: latency-oriented forward captures from vLLM/HuggingFace-style models
- `training/`: throughput-oriented training captures from PyTorch benchmark/dynamo models

Each model directory contains one subdirectory per captured fusion region. Region
directories are the data unit; they should contain exactly:

- `repro.py`: standalone module plus `make_inputs()`
- `meta.json`: hash, ops, reduction kinds, kernel count, and benchmark metadata

Region directory names follow:

```text
region_<capture_index>_<op_summary>_<graph_hash>_<shape_hash>
```

The hash is intended to make repeated captures stable enough to compare and
deduplicate. The capture index is useful for relating regions back to extraction
order, but should not be treated as a semantic identifier.

Individual `region_*` directories intentionally do not have separate READMEs;
their `meta.json` is the per-region documentation.

## Promotion Rules

Generated extraction output should first land under `output/`. Before promoting
new repros into this tree:

1. Run the new `repro.py` files through `scripts/bench.py` or at least import
   them with a CUDA-capable PyTorch build.
2. Check that `make_inputs()` preserves shape, dtype, and stride information
   from the source graph.
3. Keep only model-derived regions. Avoid hand-curated synthetic patterns unless
   the repo explicitly adds a separate synthetic corpus.
4. Delete stale versions when a model is recaptured with a more correct
   partitioner or input generator.
5. Do not commit generated logs, `output/`, `benchmark_results.json`, pyc files,
   or investigation dumps.

See the workload-specific READMEs for the currently checked-in model sets.
