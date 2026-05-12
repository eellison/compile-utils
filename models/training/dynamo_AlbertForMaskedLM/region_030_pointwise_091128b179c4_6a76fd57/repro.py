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
    def forward(self, mm_124: "f32[4096, 16384]", addmm_11: "f32[4096, 16384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        reshape_default: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_124, [8, 512, 16384]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default_1: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_11, [8, 512, 16384]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_1: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_2: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(reshape_default, add_tensor_1);  reshape_default = add_tensor_1 = None
        mul_tensor_5: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_2: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_3: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_10);  add_tensor_2 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default_2: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_tensor_3, [4096, 16384]);  add_tensor_3 = None
        permute_default: "f32[16384, 4096]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0]);  reshape_default_2 = None
        return permute_default



def make_inputs():
    return [
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
