"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_7: "f32[16384, 4096]", mm_19: "f32[16384, 4096]", mm_31: "f32[16384, 4096]", mm_43: "f32[16384, 4096]", mm_55: "f32[16384, 4096]", mm_67: "f32[16384, 4096]", mm_79: "f32[16384, 4096]", mm_91: "f32[16384, 4096]", mm_103: "f32[16384, 4096]", mm_115: "f32[16384, 4096]", mm_127: "f32[16384, 4096]", mm_139: "f32[16384, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        add_tensor: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(mm_7, mm_19);  mm_7 = mm_19 = None
        add_tensor_1: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor, mm_31);  add_tensor = mm_31 = None
        add_tensor_2: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, mm_43);  add_tensor_1 = mm_43 = None
        add_tensor_3: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_2, mm_55);  add_tensor_2 = mm_55 = None
        add_tensor_4: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_3, mm_67);  add_tensor_3 = mm_67 = None
        add_tensor_5: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_4, mm_79);  add_tensor_4 = mm_79 = None
        add_tensor_6: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_5, mm_91);  add_tensor_5 = mm_91 = None
        add_tensor_7: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_6, mm_103);  add_tensor_6 = mm_103 = None
        add_tensor_8: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_7, mm_115);  add_tensor_7 = mm_115 = None
        add_tensor_9: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_8, mm_127);  add_tensor_8 = mm_127 = None
        add_tensor_10: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_tensor_9, mm_139);  add_tensor_9 = mm_139 = None
        return add_tensor_10



def make_inputs():
    return [
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    ]
