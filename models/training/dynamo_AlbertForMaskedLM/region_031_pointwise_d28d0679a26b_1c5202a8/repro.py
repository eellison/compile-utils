"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config
from math import inf
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_5: "f32[4096, 16384]", mm_17: "f32[4096, 16384]", mm_29: "f32[4096, 16384]", mm_41: "f32[4096, 16384]", mm_53: "f32[4096, 16384]", mm_65: "f32[4096, 16384]", mm_77: "f32[4096, 16384]", mm_89: "f32[4096, 16384]", mm_101: "f32[4096, 16384]", mm_113: "f32[4096, 16384]", mm_125: "f32[4096, 16384]", mm_137: "f32[4096, 16384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        add_tensor: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(mm_5, mm_17);  mm_5 = mm_17 = None
        add_tensor_1: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor, mm_29);  add_tensor = mm_29 = None
        add_tensor_2: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_1, mm_41);  add_tensor_1 = mm_41 = None
        add_tensor_3: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_2, mm_53);  add_tensor_2 = mm_53 = None
        add_tensor_4: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_3, mm_65);  add_tensor_3 = mm_65 = None
        add_tensor_5: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_4, mm_77);  add_tensor_4 = mm_77 = None
        add_tensor_6: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_5, mm_89);  add_tensor_5 = mm_89 = None
        add_tensor_7: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_6, mm_101);  add_tensor_6 = mm_101 = None
        add_tensor_8: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_7, mm_113);  add_tensor_7 = mm_113 = None
        add_tensor_9: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_8, mm_125);  add_tensor_8 = mm_125 = None
        add_tensor_10: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_9, mm_137);  add_tensor_9 = mm_137 = None
        return add_tensor_10



def make_inputs():
    return [
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    ]


def _count_bytes(inputs, outputs):
    """Count total read + write bytes for SOL calculation."""
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


def benchmark(n_warmup=25, n_rep=200):
    from triton.testing import do_bench

    mod = Repro()
    inputs = make_inputs()

    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes = _count_bytes(inputs, eager_out)

    # SOL: memcopy same total bytes (copy half since copy does read+write)
    copy_elems = max(total_bytes // (2 * 4), 256)
    src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: dst.copy_(src), warmup=n_warmup, rep=n_rep)
    sol_us = sol_ms * 1000
    del src, dst

    # Compiled (default heuristics)
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

    print(f"\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
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
    }


if __name__ == "__main__":
    benchmark()
