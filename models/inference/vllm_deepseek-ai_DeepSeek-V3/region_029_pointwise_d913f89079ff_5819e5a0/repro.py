"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg1_1: "bf16[256, 7168]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:145 in forward, code: router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        convert_element_type_default: "f32[256, 7168]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        permute_default: "f32[7168, 256]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        return permute_default



def make_inputs():
    return [
    torch.randn([256, 7168], dtype=torch.bfloat16, device='cuda'),
    ]
