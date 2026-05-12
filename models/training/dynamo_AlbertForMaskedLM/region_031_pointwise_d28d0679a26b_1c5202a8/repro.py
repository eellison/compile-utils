"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, mm_5: "f32[4096, 16384]", mm_17: "f32[4096, 16384]", mm_29: "f32[4096, 16384]", mm_41: "f32[4096, 16384]", mm_53: "f32[4096, 16384]", mm_65: "f32[4096, 16384]", mm_77: "f32[4096, 16384]", mm_89: "f32[4096, 16384]", mm_101: "f32[4096, 16384]", mm_113: "f32[4096, 16384]", mm_125: "f32[4096, 16384]", mm_137: "f32[4096, 16384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        add_tensor: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(mm_5, mm_17);  mm_5 = mm_17 = None
        add_tensor_1: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor, mm_29);  add_tensor = mm_29 = None
        add_tensor_2: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_1, mm_41);  add_tensor_1 = mm_41 = None
        add_tensor_3: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_2, mm_53);  add_tensor_2 = mm_53 = None
        add_tensor_4: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_3, mm_65);  add_tensor_3 = mm_65 = None
        add_tensor_5: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_4, mm_77);  add_tensor_4 = mm_77 = None
        add_tensor_6: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_5, mm_89);  add_tensor_5 = mm_89 = None
        add_tensor_7: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_6, mm_101);  add_tensor_6 = mm_101 = None
        add_tensor_8: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_7, mm_113);  add_tensor_7 = mm_113 = None
        add_tensor_9: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_8, mm_125);  add_tensor_8 = mm_125 = None
        add_tensor_10: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_9, mm_137);  add_tensor_9 = mm_137 = None
        return add_tensor_10



def make_inputs():
    return [
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    ]
