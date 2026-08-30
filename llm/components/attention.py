import torch

class AttentionLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1, qkv_bias=False):
        super(AttentionLayer, self).__init__()
        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, input):
        query = self.W_Q(input)
        key = self.W_K(input)
        value = self.W_V(input)

        attn_score = torch.matmul(query, key.transpose(-2, -1))
        attn_weight = self.dropout(torch.softmax(attn_score / key.shape[-1] ** 0.5, dim=-1))
        outputs = torch.matmul(attn_weight, value)
        return outputs

