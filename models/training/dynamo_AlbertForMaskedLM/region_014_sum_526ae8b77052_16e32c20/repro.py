"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import torch
import torch._inductor.config as inductor_config
from math import inf
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_148: "f32[4096, 128]", primals_7: "f32[128]", mul: "f32[8, 512, 128]", div_39: "f32[8, 512, 1]", primals_2: "i64[1, 512]", full_default_1: "f32[]", gather: "i64[1, 512]", primals_1: "i64[8, 512]", mm_1: "f32[30000, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        reshape_default: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(mm_148, [8, 512, 128]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_7);  primals_7 = None
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_tensor_1);  div_39 = sub_tensor_1 = None
        mul_tensor_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul);  mul = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:106 in forward, code: embeddings = embeddings + position_embeddings
        sum_dim_int_list_4: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:105 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default, [primals_2], where_self, True);  full_default = primals_2 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:96 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[8, 512]" = torch.ops.aten.expand.default(gather, [8, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:102 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_4);  unsqueeze_default_1 = None
        full_default_2: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:101 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_2: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[8, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_1, mul_tensor_4);  unsqueeze_default_2 = full_default_1 = mul_tensor_4 = None
        full_default_3: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30000, 128]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_2, True);  full_default_3 = primals_1 = where_self_2 = None
        add_tensor: "f32[30000, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, add_tensor)



def make_inputs():
    return [
    torch.randn([4096, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 100, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([30000, 128], dtype=torch.float32, device='cuda'),
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
