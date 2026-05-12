"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[201088, 2880]", arg0_1: "i64[4, 512]", arg3_1: "f32[2880]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:479 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f32[4, 512, 2880]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 199999);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:59 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 2880]" = torch.ops.aten.pow.Tensor_Scalar(embedding_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:60 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:61 in forward, code: return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama
        mul_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(arg3_1, mul_tensor);  arg3_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:313 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:314 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:315 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 2880]);  mul_tensor_1 = None
        return (reshape_default, reshape_default_1, reshape_default_2)



def make_inputs():
    return [
    torch.randn([201088, 2880], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [4, 512], dtype=torch.int64, device='cuda'),
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
