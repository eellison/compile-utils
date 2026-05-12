"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=max, ranges=[], reduction_ranges=[]
#   origins: ['aten.max.dim']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, bmm_12: "f32[256, 512, 512]", where: "f32[4, 1, 512, 512]", arg61_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        reshape_default: "f32[4, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [4, 64, 512, 512]);  bmm_12 = None
        mul_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:256 in eager_attention_forward, code: attn_weights = attn_weights + causal_mask
        add_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.add.Tensor(mul_tensor, where);  mul_tensor = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:258 in eager_attention_forward, code: sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        reshape_default_1: "f32[1, 64, 1, 1]" = torch.ops.aten.reshape.default(arg61_1, [1, -1, 1, 1]);  arg61_1 = None
        expand_default: "f32[4, 64, 512, 1]" = torch.ops.aten.expand.default(reshape_default_1, [4, -1, 512, -1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:259 in eager_attention_forward, code: combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        cat_default: "f32[4, 64, 512, 513]" = torch.ops.aten.cat.default([add_tensor, expand_default], -1);  add_tensor = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        max_dim = torch.ops.aten.max.dim(cat_default, -1, True);  cat_default = None
        getitem: "f32[4, 64, 512, 1]" = max_dim[0];  max_dim = None
        return getitem



def make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]
