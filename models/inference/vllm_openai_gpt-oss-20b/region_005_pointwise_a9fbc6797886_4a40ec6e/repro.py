"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, addmm_19: "f32[2048, 32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:155 in forward, code: router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        topk_default = torch.ops.aten.topk.default(addmm_19, 4);  addmm_19 = None
        getitem: "f32[2048, 4]" = topk_default[0]
        getitem_1: "i64[2048, 4]" = topk_default[1];  topk_default = None
        return (getitem, getitem_1)



def make_inputs():
    return [
    torch.randn([2048, 32], dtype=torch.float32, device='cuda'),
    ]
