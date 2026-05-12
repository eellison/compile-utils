"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2048, 7168]", mm_2: "bf16[2048, 7168]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:206 in torch_dynamo_resume_in_forward_at_206, code: hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        reshape_default: "bf16[4, 512, 7168]" = torch.ops.aten.reshape.default(arg0_1, [4, 512, 7168]);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:105 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_1: "bf16[4, 512, 7168]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 7168]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:207 in torch_dynamo_resume_in_forward_at_206, code: hidden_states = hidden_states + self.shared_experts(residuals)
        add_tensor: "bf16[4, 512, 7168]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        return add_tensor



def make_inputs():
    return [
    torch.randn([2048, 7168], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 7168], dtype=torch.bfloat16, device='cuda'),
    ]
