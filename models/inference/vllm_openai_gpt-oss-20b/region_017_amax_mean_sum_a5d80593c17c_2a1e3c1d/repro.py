"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '2880'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=mean, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import torch
from torch import device
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg53_1: "f32[32, 2880]", bmm_11: "f32[32, 2048, 2880]", getitem_22: "f32[2048, 4]", getitem_23: "i64[2048, 4]", add_26: "f32[4, 512, 2880]", arg54_1: "f32[2880]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:136 in forward, code: next_states = next_states + self.down_proj_bias[..., None, :]
        unsqueeze_default: "f32[32, 1, 2880]" = torch.ops.aten.unsqueeze.default(arg53_1, 1);  arg53_1 = None
        add_tensor: "f32[32, 2048, 2880]" = torch.ops.aten.add.Tensor(bmm_11, unsqueeze_default);  bmm_11 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:137 in forward, code: next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        reshape_default: "f32[32, 4, 512, 2880]" = torch.ops.aten.reshape.default(add_tensor, [32, 4, -1, 2880]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:157 in forward, code: router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        full_default: "f32[2048, 32]" = torch.ops.aten.full.default([2048, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:156 in forward, code: router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(getitem_22, [1], True)
        sub_tensor: "f32[2048, 4]" = torch.ops.aten.sub.Tensor(getitem_22, amax_default);  getitem_22 = amax_default = None
        exp_default: "f32[2048, 4]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[2048, 4]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:157 in forward, code: router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        scatter_src: "f32[2048, 32]" = torch.ops.aten.scatter.src(full_default, 1, getitem_23, div_tensor);  full_default = getitem_23 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:138 in forward, code: next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        permute_default: "f32[32, 2048]" = torch.ops.aten.permute.default(scatter_src, [1, 0]);  scatter_src = None
        reshape_default_1: "f32[32, 4, 512]" = torch.ops.aten.reshape.default(permute_default, [32, 4, -1]);  permute_default = None
        unsqueeze_default_1: "f32[32, 4, 512, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, 3);  reshape_default_1 = None
        mul_tensor: "f32[32, 4, 512, 2880]" = torch.ops.aten.mul.Tensor(reshape_default, unsqueeze_default_1);  reshape_default = unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:139 in forward, code: next_states = next_states.sum(dim=0)
        sum_dim_int_list_1: "f32[4, 512, 2880]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0]);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:387 in forward, code: hidden_states = residual + hidden_states
        add_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.add.Tensor(add_26, sum_dim_int_list_1);  add_26 = sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:59 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 2880]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_1, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:60 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(add_tensor_1, rsqrt_default);  add_tensor_1 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:61 in forward, code: return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama
        mul_tensor_2: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(arg54_1, mul_tensor_1);  arg54_1 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:313 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_2, [2048, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:314 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_2, [2048, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:315 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_4: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_2, [2048, 2880]);  mul_tensor_2 = None
        return (reshape_default_2, reshape_default_3, reshape_default_4)



def make_inputs():
    return [
    torch.randn([32, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([32, 2048, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 4], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [2048, 4], dtype=torch.int64, device='cuda'),
    torch.randn([4, 512, 2880], dtype=torch.float32, device='cuda'),
    torch.randn([2880], dtype=torch.float32, device='cuda'),
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
