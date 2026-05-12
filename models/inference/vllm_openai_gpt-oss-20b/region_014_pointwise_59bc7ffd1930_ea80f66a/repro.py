"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, addmm_16: "f32[2048, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:314 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 512]);  addmm_16 = None
        reshape_default_1: "f32[4, 512, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 64]);  reshape_default = None
        permute_default: "f32[4, 8, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:227 in _apply_rotary_emb, code: first_half, second_half = torch.chunk(x, 2, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(permute_default, 32, -1);  permute_default = None
        getitem: "f32[4, 8, 512, 32]" = split_tensor[0]
        getitem_1: "f32[4, 8, 512, 32]" = split_tensor[1];  split_tensor = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    ]
