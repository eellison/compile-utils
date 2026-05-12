"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg68_1: "f32[32, 5760]", bmm_14: "f32[32, 2048, 5760]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:130 in forward, code: gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
        unsqueeze_default: "f32[32, 1, 5760]" = torch.ops.aten.unsqueeze.default(arg68_1, 1);  arg68_1 = None
        add_tensor: "f32[32, 2048, 5760]" = torch.ops.aten.add.Tensor(bmm_14, unsqueeze_default);  bmm_14 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:131 in forward, code: gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        slice_tensor: "f32[32, 2048, 2880]" = torch.ops.aten.slice.Tensor(add_tensor, 2, 1, 9223372036854775807, 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:133 in forward, code: up = up.clamp(min=-self.limit, max=self.limit)
        clamp_min_default: "f32[32, 2048, 2880]" = torch.ops.aten.clamp_min.default(slice_tensor, -7.0);  slice_tensor = None
        clamp_max_default: "f32[32, 2048, 2880]" = torch.ops.aten.clamp_max.default(clamp_min_default, 7.0);  clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:135 in forward, code: next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        add_tensor_1: "f32[32, 2048, 2880]" = torch.ops.aten.add.Tensor(clamp_max_default, 1);  clamp_max_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:131 in forward, code: gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        slice_tensor_1: "f32[32, 2048, 2880]" = torch.ops.aten.slice.Tensor(add_tensor, 2, 0, 9223372036854775807, 2);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:132 in forward, code: gate = gate.clamp(min=None, max=self.limit)
        clamp_max_default_1: "f32[32, 2048, 2880]" = torch.ops.aten.clamp_max.default(slice_tensor_1, 7.0);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:134 in forward, code: glu = gate * torch.sigmoid(gate * self.alpha)
        mul_tensor: "f32[32, 2048, 2880]" = torch.ops.aten.mul.Tensor(clamp_max_default_1, 1.702)
        sigmoid_default: "f32[32, 2048, 2880]" = torch.ops.aten.sigmoid.default(mul_tensor);  mul_tensor = None
        mul_tensor_1: "f32[32, 2048, 2880]" = torch.ops.aten.mul.Tensor(clamp_max_default_1, sigmoid_default);  clamp_max_default_1 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:135 in forward, code: next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        mul_tensor_2: "f32[32, 2048, 2880]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_1);  add_tensor_1 = mul_tensor_1 = None
        return mul_tensor_2



def make_inputs():
    return [
    torch.randn([32, 5760], dtype=torch.float32, device='cuda'),
    torch.randn([32, 2048, 5760], dtype=torch.float32, device='cuda'),
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
