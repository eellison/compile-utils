"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_7: "bf16[2048, 7168]", add_5: "bf16[4, 512, 7168]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:105 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 7168]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 7168]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:495 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "bf16[4, 512, 7168]" = torch.ops.aten.add.Tensor(add_5, reshape_default);  add_5 = reshape_default = None
        return add_tensor



def make_inputs():
    return [
    torch.randn([2048, 7168], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 7168], dtype=torch.bfloat16, device='cuda'),
    ]
