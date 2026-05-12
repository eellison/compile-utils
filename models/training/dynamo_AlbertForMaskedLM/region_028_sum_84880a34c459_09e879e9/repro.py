"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import torch
import torch._inductor.config as inductor_config
from math import inf
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_126: "f32[4096, 4096]", mul_411: "f32[8, 512, 4096]", primals_19: "f32[4096]", mul_14: "f32[8, 512, 4096]", div_36: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_126, [8, 512, 4096]);  mm_126 = None
        add_tensor: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_411, reshape_default);  mul_411 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor, primals_19);  add_tensor = primals_19 = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_14);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_14, sum_dim_int_list_1);  mul_14 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_36, sub_tensor_1);  div_36 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_1: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_tensor_4, [4096, 4096]);  mul_tensor_4 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0]);  reshape_default_1 = None
        return permute_default



def make_inputs():
    return [
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
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
