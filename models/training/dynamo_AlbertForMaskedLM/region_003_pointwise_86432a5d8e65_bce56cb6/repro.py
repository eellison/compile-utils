"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, add_111: "f32[8, 512, 4096]", getitem_49: "f32[8, 512, 1]", getitem_48: "f32[8, 512, 1]", arg24_1: "f32[4096]", arg25_1: "f32[4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_111, getitem_49);  add_111 = getitem_49 = None
        add_tensor: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg24_1);  mul_tensor = arg24_1 = None
        add_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg25_1);  mul_tensor_1 = arg25_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, [4096, 4096]);  add_tensor_1 = None
        return reshape_default



def make_inputs():
    return [
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    ]
