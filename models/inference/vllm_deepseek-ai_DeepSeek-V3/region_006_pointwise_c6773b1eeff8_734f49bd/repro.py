"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_5: "bf16[2048, 18432]", mm_6: "bf16[2048, 18432]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:105 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 18432]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 18432]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[4, 512, 18432]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        neg_default: "f32[4, 512, 18432]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[4, 512, 18432]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 18432]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 18432]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[4, 512, 18432]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:105 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_1: "bf16[4, 512, 18432]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 18432]);  mm_6 = None
        mul_tensor: "bf16[4, 512, 18432]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, reshape_default_1);  convert_element_type_default_1 = reshape_default_1 = None
        reshape_default_2: "bf16[2048, 18432]" = torch.ops.aten.reshape.default(mul_tensor, [2048, 18432]);  mul_tensor = None
        return reshape_default_2



def make_inputs():
    return [
    torch.randn([2048, 18432], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 18432], dtype=torch.bfloat16, device='cuda'),
    ]
