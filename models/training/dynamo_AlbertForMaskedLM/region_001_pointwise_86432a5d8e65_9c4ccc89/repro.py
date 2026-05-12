"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mul_125: "f32[8, 512, 128]", getitem_51: "f32[8, 512, 1]", getitem_50: "f32[8, 512, 1]", arg28_1: "f32[128]", arg29_1: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_125, getitem_51);  mul_125 = getitem_51 = None
        add_tensor: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg28_1);  mul_tensor = arg28_1 = None
        add_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg29_1);  mul_tensor_1 = arg29_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_tensor_1, [4096, 128]);  add_tensor_1 = None
        return reshape_default



def make_inputs():
    return [
    torch.randn([8, 512, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    ]
