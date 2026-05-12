"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, sigmoid: "f32[2048, 256]", getitem_5: "i64[2048, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:148 in forward, code: topk_weights = scores.gather(1, topk_indices)
        gather_default: "f32[2048, 8]" = torch.ops.aten.gather.default(sigmoid, 1, getitem_5);  sigmoid = getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:150 in forward, code: denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(gather_default, [-1], True)
        add_tensor: "f32[2048, 1]" = torch.ops.aten.add.Tensor(sum_dim_int_list, 1e-20);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:151 in forward, code: topk_weights /= denominator
        div_tensor: "f32[2048, 8]" = torch.ops.aten.div.Tensor(gather_default, add_tensor);  gather_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:152 in forward, code: topk_weights = topk_weights * self.routed_scaling_factor
        mul_tensor: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(div_tensor, 2.5);  div_tensor = None
        return mul_tensor



def make_inputs():
    return [
    torch.randn([2048, 256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [2048, 8], dtype=torch.int64, device='cuda'),
    ]
