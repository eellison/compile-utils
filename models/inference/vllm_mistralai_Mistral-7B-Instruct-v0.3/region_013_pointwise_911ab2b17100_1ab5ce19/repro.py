"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
from torch import device
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, iota: "i64[512]", cumsum: "i64[4, 512]"):
        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[512, 1]" = torch.ops.aten.full.default([512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default: "i64[512, 1]" = torch.ops.aten.reshape.default(iota, [512, 1])
        le_tensor: "b8[512, 512]" = torch.ops.aten.le.Tensor(add_tensor, reshape_default);  reshape_default = None
        bitwise_and_tensor: "b8[512, 512]" = torch.ops.aten.bitwise_and.Tensor(full_default, le_tensor);  full_default = le_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:386 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_default_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_1: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_1, [4, 1])
        index_tensor: "i64[4, 512]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_1, iota]);  reshape_default_1 = iota = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_2: "i64[4, 512, 1]" = torch.ops.aten.reshape.default(index_tensor, [4, 512, 1]);  index_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        reshape_default_3: "i64[4, 1]" = torch.ops.aten.reshape.default(iota_default_1, [4, 1]);  iota_default_1 = None
        index_tensor_1: "i64[4, 512]" = torch.ops.aten.index.Tensor(cumsum, [reshape_default_3, add_tensor]);  cumsum = reshape_default_3 = add_tensor = None

        # File: /tmp/pytorch-work/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        reshape_default_4: "i64[4, 1, 512]" = torch.ops.aten.reshape.default(index_tensor_1, [4, 1, 512]);  index_tensor_1 = None
        eq_tensor: "b8[4, 512, 512]" = torch.ops.aten.eq.Tensor(reshape_default_2, reshape_default_4);  reshape_default_2 = reshape_default_4 = None
        bitwise_and_tensor_1: "b8[4, 512, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, eq_tensor);  bitwise_and_tensor = eq_tensor = None

        # File: /tmp/pytorch-work/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        reshape_default_5: "b8[4, 1, 512, 512]" = torch.ops.aten.reshape.default(bitwise_and_tensor_1, [4, 1, 512, 512]);  bitwise_and_tensor_1 = None
        expand_default: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(reshape_default_5, [4, 1, 512, 512]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default_2);  full_default_1 = full_default_2 = None
        full_default_3: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_3, full_default_4);  full_default_3 = full_default_4 = None
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_2: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_5, full_default_6);  full_default_5 = full_default_6 = None
        full_default_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_7, full_default_8);  full_default_7 = full_default_8 = None
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_9, full_default_10);  full_default_9 = full_default_10 = None
        full_default_11: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_5: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_11, full_default_12);  full_default_11 = full_default_12 = None
        full_default_13: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_13, full_default_14);  full_default_13 = full_default_14 = None
        full_default_15: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "bf16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_default, full_default_15, full_default_16);  expand_default = full_default_15 = full_default_16 = None
        return (where_self, where_self_1, where_self_2, where_self_3, where_self_4, where_self_5, where_self_6, where_self_7)



def make_inputs():
    return [
    torch.randint(0, 100, [512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 100, [4, 512], dtype=torch.int64, device='cuda'),
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
