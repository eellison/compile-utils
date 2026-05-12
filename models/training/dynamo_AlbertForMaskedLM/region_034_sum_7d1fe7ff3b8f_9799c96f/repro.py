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
#   type=sum, ranges=['1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import torch
import torch._inductor.config as inductor_config
from math import inf
from torch import device

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm: "f32[4096, 128]", primals_29: "f32[128]", addmm_73: "f32[4096, 128]", getitem_51: "f32[8, 512, 1]", rsqrt_25: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(mm, [8, 512, 128]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_29);  primals_29 = None
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [8, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        pow_tensor_scalar: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_5, getitem_51);  mul_tensor_5 = getitem_51 = None
        mul_tensor_6: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = None
        mul_tensor_7: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_6);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True);  mul_tensor_7 = None
        mul_tensor_8: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_8);  sub_tensor_1 = mul_tensor_8 = None
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_tensor_9: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_10: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor_6);  mul_tensor_6 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_11: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_12: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, add_tensor_1);  mul_tensor_9 = add_tensor_1 = None
        mul_tensor_13: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_tensor_13);  mul_tensor_13 = None
        mul_tensor_14: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_11, sub_tensor_3);  mul_tensor_11 = sub_tensor_3 = None
        mul_tensor_15: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_14, 0.7978845608028654);  mul_tensor_14 = None
        mul_tensor_16: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_15, 0.044715)
        pow_tensor_scalar_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[8, 512, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_17: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_16, mul_scalar);  mul_tensor_16 = mul_scalar = None
        add_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_17);  mul_tensor_15 = mul_tensor_17 = None
        mul_tensor_18: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.5);  mul_tensor_12 = None
        add_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_18);  add_tensor_2 = mul_tensor_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_tensor_3, [4096, 128]);  add_tensor_3 = None
        permute_default: "f32[128, 4096]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [128]);  sum_dim_int_list_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, reshape_default_3)



def make_inputs():
    return [
    torch.randn([4096, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
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
