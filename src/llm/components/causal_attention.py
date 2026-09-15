import torch


class CausalAttentionLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input):
        query = self.W_Q(input)
        key = self.W_K(input)
        value = self.W_V(input)

        attn_score = torch.matmul(query, key.transpose(-2, -1))
        attn_score = self.dropout(
            attn_score.masked_fill(
                self.mask.bool()[: input.shape[-2], : input.shape[-2]], -float("inf")
            )
        )
        attn_weight = torch.softmax(attn_score / input.shape[-1] ** 0.5, dim=-1)
        outputs = torch.matmul(attn_weight, value)
        return outputs

