"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 512, 7168]", arg1_1: "bf16[4, 512, 7168]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:495 in torch_dynamo_resume_in_forward_at_494, code: hidden_states = residual + hidden_states
        add_tensor: "bf16[4, 512, 7168]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return add_tensor



def make_inputs():
    return [
    torch.randn([4, 512, 7168], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 7168], dtype=torch.bfloat16, device='cuda'),
    ]
