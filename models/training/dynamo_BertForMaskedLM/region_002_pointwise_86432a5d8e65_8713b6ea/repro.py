"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mul_112: "f32[32, 512, 768]", getitem_51: "f32[32, 512, 1]", getitem_50: "f32[32, 512, 1]", arg202_1: "f32[768]", arg203_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_112, getitem_51);  mul_112 = getitem_51 = None
        add_tensor: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg202_1);  mul_tensor = arg202_1 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg203_1);  mul_tensor_1 = arg203_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [16384, 768]);  add_tensor_1 = None
        return reshape_default



def make_inputs():
    return [
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]
