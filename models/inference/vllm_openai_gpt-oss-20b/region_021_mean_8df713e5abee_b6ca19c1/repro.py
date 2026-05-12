"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=mean, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.mean.dim']
"""
import torch
import torch._inductor.config as inductor_config

# The extracted FX graph subgraph:
class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[201088, 2880]", arg0_1: "i64[4, 512]", arg3_1: "f32[2880]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:479 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f32[4, 512, 2880]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 199999);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:59 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 2880]" = torch.ops.aten.pow.Tensor_Scalar(embedding_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:60 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:61 in forward, code: return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama
        mul_tensor_1: "f32[4, 512, 2880]" = torch.ops.aten.mul.Tensor(arg3_1, mul_tensor);  arg3_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:313 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:314 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 2880])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:315 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[2048, 2880]" = torch.ops.aten.reshape.default(mul_tensor_1, [2048, 2880]);  mul_tensor_1 = None
        return (reshape_default, reshape_default_1, reshape_default_2)



def make_inputs():
    return [
    torch.randn([201088, 2880], dtype=torch.float32, device='cuda'),
    torch.randint(0, 100, [4, 512], dtype=torch.int64, device='cuda'),
    torch.randn([2880], dtype=torch.float32, device='cuda'),
    ]
