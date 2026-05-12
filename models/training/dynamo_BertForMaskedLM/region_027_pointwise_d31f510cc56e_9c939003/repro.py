"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, primals_4: "f32[30522, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_default: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_1: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        constant_pad_nd_default: "f32[30524, 768]" = torch.ops.aten.constant_pad_nd.default(permute_default_1, [0, 0, 0, 2]);  permute_default_1 = None
        return constant_pad_nd_default



def make_inputs():
    return [
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    ]
