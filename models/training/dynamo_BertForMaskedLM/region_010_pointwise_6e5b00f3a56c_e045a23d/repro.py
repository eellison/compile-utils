"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, addmm_67: "f32[16384, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        reshape_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, [32, 512, -1, 64]);  reshape_default = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_1: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_default, [0, 1, 3, 2]);  permute_default = None
        mul_scalar: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_1, 0.3535533905932738);  permute_default_1 = None
        expand_default: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar, [32, 12, 64, 512]);  mul_scalar = None
        clone_default: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_default, [384, 64, 512]);  clone_default = None
        return reshape_default_2



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    ]
