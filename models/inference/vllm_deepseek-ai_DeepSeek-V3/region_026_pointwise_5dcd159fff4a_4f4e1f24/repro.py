"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, getitem_3: "i64[2048, 4]", add: "f32[2048, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:132 in get_topk_indices, code: group_mask = torch.zeros_like(group_scores)
        full_default: "f32[2048, 8]" = torch.ops.aten.full.default([2048, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:133 in get_topk_indices, code: group_mask.scatter_(1, group_idx, 1)
        scatter_value: "f32[2048, 8]" = torch.ops.aten.scatter.value(full_default, 1, getitem_3, 1);  full_default = getitem_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:136 in get_topk_indices, code: .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
        unsqueeze_default: "f32[2048, 8, 1]" = torch.ops.aten.unsqueeze.default(scatter_value, -1);  scatter_value = None
        expand_default: "f32[2048, 8, 32]" = torch.ops.aten.expand.default(unsqueeze_default, [-1, 8, 32]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:137 in get_topk_indices, code: .reshape(-1, self.n_routed_experts)
        clone_default: "f32[2048, 8, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default: "f32[2048, 256]" = torch.ops.aten.reshape.default(clone_default, [2048, 256]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:139 in get_topk_indices, code: scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        convert_element_type_default: "b8[2048, 256]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.bool);  reshape_default = None
        bitwise_not_default: "b8[2048, 256]" = torch.ops.aten.bitwise_not.default(convert_element_type_default);  convert_element_type_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[2048, 256]" = torch.ops.aten.where.self(bitwise_not_default, full_default_1, add);  bitwise_not_default = full_default_1 = add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:140 in get_topk_indices, code: topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_default = torch.ops.aten.topk.default(where_self, 8, -1, True, False);  where_self = None
        getitem: "i64[2048, 8]" = topk_default[1];  topk_default = None
        return getitem



def make_inputs():
    return [
    torch.randint(0, 100, [2048, 4], dtype=torch.int64, device='cuda'),
    torch.randn([2048, 256], dtype=torch.float32, device='cuda'),
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
