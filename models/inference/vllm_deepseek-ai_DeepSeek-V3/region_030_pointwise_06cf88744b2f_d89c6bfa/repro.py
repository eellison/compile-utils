"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 512, 7168]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:144 in forward, code: hidden_states = hidden_states.view(-1, self.config.hidden_size)
        reshape_default: "bf16[2048, 7168]" = torch.ops.aten.reshape.default(arg0_1, [-1, 7168]);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:145 in forward, code: router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        convert_element_type_default: "f32[2048, 7168]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        return convert_element_type_default



def make_inputs():
    return [
    torch.randn([4, 512, 7168], dtype=torch.bfloat16, device='cuda'),
    ]
