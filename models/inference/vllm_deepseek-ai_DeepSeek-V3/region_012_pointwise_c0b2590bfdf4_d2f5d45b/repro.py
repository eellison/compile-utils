"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, getitem_1: "bf16[4, 128, 512, 64]", arg8_1: "bf16[1, 512, 64]", arg9_1: "bf16[1, 512, 64]", getitem: "bf16[4, 128, 512, 128]", getitem_3: "bf16[4, 512, 64]", getitem_4: "bf16[4, 128, 512, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:311 in apply_rotary_pos_emb_interleave, code: q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_default: "bf16[4, 128, 512, 32, 2]" = torch.ops.aten.reshape.default(getitem_1, [4, 128, 512, 32, 2]);  getitem_1 = None
        permute_default: "bf16[4, 128, 512, 2, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        clone_default: "bf16[4, 128, 512, 2, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "bf16[4, 128, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [4, 128, 512, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:307 in apply_rotary_pos_emb_interleave, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:316 in apply_rotary_pos_emb_interleave, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor: "bf16[4, 128, 512, 64]" = torch.ops.aten.mul.Tensor(reshape_default_1, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:214 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 128, 512, 32]" = torch.ops.aten.slice.Tensor(reshape_default_1, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:215 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 128, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:213 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 128, 512, 32]" = torch.ops.aten.slice.Tensor(reshape_default_1, 3, 0, 32);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:215 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 128, 512, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:308 in apply_rotary_pos_emb_interleave, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "bf16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(arg9_1, 1);  arg9_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:316 in apply_rotary_pos_emb_interleave, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_1: "bf16[4, 128, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "bf16[4, 128, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:415 in forward, code: query_states = torch.cat((q_pass, q_rot), dim=-1)
        cat_default_1: "bf16[4, 128, 512, 192]" = torch.ops.aten.cat.default([getitem, add_tensor], -1);  getitem = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:406 in forward, code: k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        reshape_default_2: "bf16[4, 1, 512, 64]" = torch.ops.aten.reshape.default(getitem_3, [4, 1, 512, 64]);  getitem_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:314 in apply_rotary_pos_emb_interleave, code: k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_default_3: "bf16[4, 1, 512, 32, 2]" = torch.ops.aten.reshape.default(reshape_default_2, [4, 1, 512, 32, 2]);  reshape_default_2 = None
        permute_default_1: "bf16[4, 1, 512, 2, 32]" = torch.ops.aten.permute.default(reshape_default_3, [0, 1, 2, 4, 3]);  reshape_default_3 = None
        clone_default_1: "bf16[4, 1, 512, 2, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "bf16[4, 1, 512, 64]" = torch.ops.aten.reshape.default(clone_default_1, [4, 1, 512, 64]);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:317 in apply_rotary_pos_emb_interleave, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "bf16[4, 1, 512, 64]" = torch.ops.aten.mul.Tensor(reshape_default_4, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:214 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "bf16[4, 1, 512, 32]" = torch.ops.aten.slice.Tensor(reshape_default_4, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:215 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "bf16[4, 1, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:213 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "bf16[4, 1, 512, 32]" = torch.ops.aten.slice.Tensor(reshape_default_4, 3, 0, 32);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:215 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_2: "bf16[4, 1, 512, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:317 in apply_rotary_pos_emb_interleave, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_3: "bf16[4, 1, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default_2, unsqueeze_default_1);  cat_default_2 = unsqueeze_default_1 = None
        add_tensor_1: "bf16[4, 1, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:413 in forward, code: k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        expand_default: "bf16[4, 128, 512, 64]" = torch.ops.aten.expand.default(add_tensor_1, [4, 128, 512, -1]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:416 in forward, code: key_states = torch.cat((k_pass, k_rot), dim=-1)
        cat_default_3: "bf16[4, 128, 512, 192]" = torch.ops.aten.cat.default([getitem_4, expand_default], -1);  getitem_4 = expand_default = None
        return (cat_default_1, cat_default_3)



def make_inputs():
    return [
    torch.randn([4, 128, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 128, 512, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 128, 512, 128], dtype=torch.bfloat16, device='cuda'),
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
