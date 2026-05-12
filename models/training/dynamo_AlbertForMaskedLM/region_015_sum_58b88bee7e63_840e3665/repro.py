"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_142: "f32[4096, 4096]", mul_453: "f32[8, 512, 4096]", mm_144: "f32[4096, 4096]", mm_146: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_142, [8, 512, 4096]);  mm_142 = None
        add_tensor: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_453, reshape_default);  mul_453 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_144, [8, 512, 4096]);  mm_144 = None
        add_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_146, [8, 512, 4096]);  mm_146 = None
        add_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        reshape_default_3: "f32[4096, 4096]" = torch.ops.aten.reshape.default(add_tensor_2, [4096, 4096]);  add_tensor_2 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0])
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(reshape_default_3, [0], True);  reshape_default_3 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list, [4096]);  sum_dim_int_list = None
        return (permute_default, reshape_default_4)



def make_inputs():
    return [
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]
