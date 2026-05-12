"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_49: "bf16[2048, 4096]", convert_element_type_1: "bf16[1, 512, 128]", convert_element_type_2: "bf16[1, 512, 128]", mm_50: "bf16[2048, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:153 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_49, [4, 512, 4096]);  mm_49 = None
        reshape_default_1: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 128]);  reshape_default = None
        permute_default: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_1: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:154 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 1024]);  mm_50 = None
        reshape_default_3: "bf16[4, 512, 8, 128]" = torch.ops.aten.reshape.default(reshape_default_2, [4, 512, -1, 128]);  reshape_default_2 = None
        permute_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:81 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:81 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_2: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        expand_default: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.expand.default(unsqueeze_default_2, [4, 8, 4, 512, 128]);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "bf16[4, 32, 512, 128]" = torch.ops.aten.reshape.default(clone_default, [4, 32, 512, 128]);  clone_default = None
        return (add_tensor, reshape_default_4)



def make_inputs():
    return [
    torch.randn([2048, 4096], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
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
