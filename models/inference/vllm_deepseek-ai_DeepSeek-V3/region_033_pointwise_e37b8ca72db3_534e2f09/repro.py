"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm: "bf16[61, 2048]", mm_1: "bf16[61, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[61, 2048]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg_default: "f32[61, 2048]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[61, 2048]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[61, 2048]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[61, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[61, 2048]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:105 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_tensor: "bf16[61, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mm_1);  convert_element_type_default_1 = mm_1 = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([61, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([61, 2048], dtype=torch.bfloat16, device='cuda'),
    ]
