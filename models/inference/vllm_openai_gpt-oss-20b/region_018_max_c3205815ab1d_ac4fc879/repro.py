"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=max, ranges=[], reduction_ranges=[]
#   origins: ['aten.max.dim']
"""
import torch
from torch import device
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, bmm_4: "f32[256, 512, 512]", iota: "i64[512]", arg27_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        reshape_default: "f32[4, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [4, 64, 512, 512]);  bmm_4 = None
        mul_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_1: "i64[512, 1]" = torch.ops.aten.reshape.default(iota, [512, 1]);  iota = None
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        expand_default: "b8[1, 512, 512]" = torch.ops.aten.expand.default(le_tensor, [1, 512, 512]);  le_tensor = None
        expand_default_1: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(expand_default, [4, 1, 512, 512]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:521 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default_1, full_default, full_default_1);  expand_default_1 = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:256 in eager_attention_forward, code: attn_weights = attn_weights + causal_mask
        add_tensor_1: "f32[4, 64, 512, 512]" = torch.ops.aten.add.Tensor(mul_tensor, where_self);  mul_tensor = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:258 in eager_attention_forward, code: sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        reshape_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.reshape.default(arg27_1, [1, -1, 1, 1]);  arg27_1 = None
        expand_default_2: "f32[4, 64, 512, 1]" = torch.ops.aten.expand.default(reshape_default_2, [4, -1, 512, -1]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:259 in eager_attention_forward, code: combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        cat_default: "f32[4, 64, 512, 513]" = torch.ops.aten.cat.default([add_tensor_1, expand_default_2], -1);  add_tensor_1 = expand_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        max_dim = torch.ops.aten.max.dim(cat_default, -1, True);  cat_default = None
        getitem: "f32[4, 64, 512, 1]" = max_dim[0];  max_dim = None
        return getitem



def make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [512], dtype=torch.int64, device='cuda'),
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
