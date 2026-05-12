"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_1: "bf16[2048, 24576]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:396 in forward, code: q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        reshape_default: "bf16[4, 512, 24576]" = torch.ops.aten.reshape.default(mm_1, [4, 512, 24576]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:397 in forward, code: q_states = q_states.view(query_shape).transpose(1, 2)
        reshape_default_1: "bf16[4, 512, 128, 192]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 192]);  reshape_default = None
        permute_default: "bf16[4, 128, 512, 192]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:398 in forward, code: q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(permute_default, [128, 64], -1);  permute_default = None
        getitem: "bf16[4, 128, 512, 128]" = split_with_sizes_default[0]
        getitem_1: "bf16[4, 128, 512, 64]" = split_with_sizes_default[1];  split_with_sizes_default = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([2048, 24576], dtype=torch.bfloat16, device='cuda'),
    ]
