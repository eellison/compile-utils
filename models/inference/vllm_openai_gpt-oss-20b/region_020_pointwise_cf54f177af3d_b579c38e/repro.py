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
    def forward(self, arg2_1: "f32[32]", getitem: "f32[4, 64, 512, 32]", getitem_1: "f32[4, 64, 512, 32]", getitem_2: "f32[4, 8, 512, 32]", getitem_3: "f32[4, 8, 512, 32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:197 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_default: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default: "f32[1, 32, 1]" = torch.ops.aten.expand.default(unsqueeze_default_1, [1, -1, 1]);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:202 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_1: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand_default, [1, 32, 1]);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:483 in forward, code: cache_position = torch.arange(
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:487 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze_default_2: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:198 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        unsqueeze_default_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1);  unsqueeze_default_2 = None
        convert_element_type_default: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_3, torch.float32);  unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:202 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default, [1, 1, 512]);  convert_element_type_default = None
        mul_tensor: "f32[1, 32, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default: "f32[1, 512, 32]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:204 in forward, code: cos = emb.cos() * self.attention_scaling
        cos_default: "f32[1, 512, 32]" = torch.ops.aten.cos.default(permute_default)
        mul_tensor_1: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(cos_default, 1.3465735902799727);  cos_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:234 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default_4: "f32[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 1);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_2: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:205 in forward, code: sin = emb.sin() * self.attention_scaling
        sin_default: "f32[1, 512, 32]" = torch.ops.aten.sin.default(permute_default);  permute_default = None
        mul_tensor_3: "f32[1, 512, 32]" = torch.ops.aten.mul.Tensor(sin_default, 1.3465735902799727);  sin_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:235 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_5: "f32[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 1);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_4: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_default_5)
        sub_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:229 in _apply_rotary_emb, code: second_ = second_half * cos + first_half * sin
        mul_tensor_5: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_default_4);  getitem_1 = None
        mul_tensor_6: "f32[4, 64, 512, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default_5);  getitem = None
        add_tensor: "f32[4, 64, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:230 in _apply_rotary_emb, code: return torch.cat((first_, second_), dim=-1)
        cat_default: "f32[4, 64, 512, 64]" = torch.ops.aten.cat.default([sub_tensor, add_tensor], -1);  sub_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        expand_default_3: "f32[4, 64, 512, 64]" = torch.ops.aten.expand.default(cat_default, [4, 64, 512, 64]);  cat_default = None
        reshape_default: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(expand_default_3, [256, 512, 64]);  expand_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:228 in _apply_rotary_emb, code: first_ = first_half * cos - second_half * sin
        mul_tensor_7: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_default_4)
        mul_tensor_8: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_default_5)
        sub_tensor_1: "f32[4, 8, 512, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:229 in _apply_rotary_emb, code: second_ = second_half * cos + first_half * sin
        mul_tensor_9: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_default_4);  getitem_3 = unsqueeze_default_4 = None
        mul_tensor_10: "f32[4, 8, 512, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_default_5);  getitem_2 = unsqueeze_default_5 = None
        add_tensor_1: "f32[4, 8, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_10);  mul_tensor_9 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:230 in _apply_rotary_emb, code: return torch.cat((first_, second_), dim=-1)
        cat_default_1: "f32[4, 8, 512, 64]" = torch.ops.aten.cat.default([sub_tensor_1, add_tensor_1], -1);  sub_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:218 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_6: "f32[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(cat_default_1, 2);  cat_default_1 = None
        expand_default_4: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_6, [4, 8, 8, 512, 64]);  unsqueeze_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:219 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.clone.default(expand_default_4, memory_format = torch.contiguous_format);  expand_default_4 = None
        reshape_default_1: "f32[4, 64, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [4, 64, 512, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        permute_default_1: "f32[4, 64, 64, 512]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        expand_default_5: "f32[4, 64, 64, 512]" = torch.ops.aten.expand.default(permute_default_1, [4, 64, 64, 512]);  permute_default_1 = None
        reshape_default_2: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(expand_default_5, [256, 64, 512]);  expand_default_5 = None
        return (reshape_default, reshape_default_2)



def make_inputs():
    return [
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 64, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 512, 32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 512, 32], dtype=torch.float32, device='cuda'),
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
