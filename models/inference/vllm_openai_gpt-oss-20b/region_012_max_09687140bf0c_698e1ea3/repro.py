"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=max, ranges=[], reduction_ranges=[]
#   origins: ['aten.max.dim']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, bmm_12: "f32[256, 512, 512]", where: "f32[4, 1, 512, 512]", arg61_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        reshape_default: "f32[4, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [4, 64, 512, 512]);  bmm_12 = None
        mul_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:256 in eager_attention_forward, code: attn_weights = attn_weights + causal_mask
        add_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.add.Tensor(mul_tensor, where);  mul_tensor = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:258 in eager_attention_forward, code: sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        reshape_default_1: "f32[1, 64, 1, 1]" = torch.ops.aten.reshape.default(arg61_1, [1, -1, 1, 1]);  arg61_1 = None
        expand_default: "f32[4, 64, 512, 1]" = torch.ops.aten.expand.default(reshape_default_1, [4, -1, 512, -1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:259 in eager_attention_forward, code: combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        cat_default: "f32[4, 64, 512, 513]" = torch.ops.aten.cat.default([add_tensor, expand_default], -1);  add_tensor = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        max_dim = torch.ops.aten.max.dim(cat_default, -1, True);  cat_default = None
        getitem: "f32[4, 64, 512, 1]" = max_dim[0];  max_dim = None
        return getitem



def make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
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
