"""
Benchmark runner for extracted kernel repros.

Usage:
    python scripts/bench.py models/inference/vllm_openai_gpt-oss-20b/region_011_.../repro.py
    python scripts/bench.py models/inference/vllm_openai_gpt-oss-20b/  # all in dir
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from triton.testing import do_bench


def _count_bytes(inputs, outputs):
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    if isinstance(outputs, torch.Tensor):
        total += outputs.nelement() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                total += o.nelement() * o.element_size()
    return total


def _count_kernels(mod, inputs):
    from torch._inductor.utils import fresh_inductor_cache
    from torch._inductor.codecache import cache_dir

    torch._dynamo.reset()
    with fresh_inductor_cache():
        compiled = torch.compile(mod)
        with torch.no_grad():
            compiled(*inputs)
            torch.cuda.synchronize()
        cd = cache_dir()
        py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
        for f in reversed(py_files):
            with open(f) as fh:
                content = fh.read()
            if 'def call(' in content and '.run(' in content:
                runs = [l for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
                return len(runs)
    return 0


def load_repro(path):
    spec = importlib.util.spec_from_file_location("repro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Repro(), mod.make_inputs()


def benchmark_one(repro_path, n_warmup=25, n_rep=200):
    mod, inputs = load_repro(repro_path)

    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes = _count_bytes(inputs, eager_out)

    # Count kernels
    n_kernels = _count_kernels(mod, inputs)

    # SOL: memcopy same total bytes
    copy_elems = max(total_bytes // (2 * 4), 256)
    src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: dst.copy_(src), warmup=n_warmup, rep=n_rep)
    sol_us = sol_ms * 1000
    del src, dst

    # Compiled (default heuristics)
    torch._dynamo.reset()
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    compiled_ms = do_bench(lambda: compiled(*inputs), warmup=n_warmup, rep=n_rep)
    compiled_us = compiled_ms * 1000

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled_cd(*inputs)
        torch.cuda.synchronize()
    cd_ms = do_bench(lambda: compiled_cd(*inputs), warmup=n_warmup, rep=n_rep)
    cd_us = cd_ms * 1000
    inductor_config.coordinate_descent_tuning = False

    print(f"\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
    print(f"Kernels generated: {n_kernels}")
    print(f"Memcopy SOL (same size): {sol_us:8.1f} us")
    print(f"Compiled (default):      {compiled_us:8.1f} us")
    print(f"Compiled (coord desc):   {cd_us:8.1f} us")
    print(f"Gap (default / SOL):     {compiled_us / sol_us:8.2f}x")
    print(f"Gap (CD / SOL):          {cd_us / sol_us:8.2f}x")

    return {
        "compiled_us": compiled_us,
        "coord_descent_us": cd_us,
        "memcopy_sol_us": sol_us,
        "total_bytes": total_bytes,
        "n_kernels": n_kernels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="repro.py file or directory containing regions")
    parser.add_argument("--update-meta", action="store_true", help="Write results to meta.json")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        result = benchmark_one(args.path)
        if args.update_meta:
            meta_path = os.path.join(os.path.dirname(args.path), "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                meta.setdefault("perf", {})["B200"] = {
                    "compiled_us": round(result["compiled_us"], 1),
                    "coord_descent_us": round(result["coord_descent_us"], 1),
                    "memcpy_sol_us": round(result["memcopy_sol_us"], 1),
                    "total_bytes": result["total_bytes"],
                }
                meta["num_kernels"] = result["n_kernels"]
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
    else:
        repros = sorted(glob.glob(os.path.join(args.path, "**/repro.py"), recursive=True))
        print(f"Found {len(repros)} repros in {args.path}\n")
        for repro in repros:
            rel = os.path.relpath(os.path.dirname(repro), args.path)
            print(f"--- {rel} ---")
            try:
                benchmark_one(repro)
            except Exception as e:
                print(f"  FAILED: {e}")
            print()


if __name__ == "__main__":
    main()
