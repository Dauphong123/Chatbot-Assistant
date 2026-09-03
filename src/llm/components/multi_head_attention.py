import torch

class MultiheadAttentionLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out muss be divisible with num_head"
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        ) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)

        # seperate the dimension to smaller num_heads with head_dim dimension
        query = query.view(b, num_tokens, self.num_heads, self.head_dim)
        key = key.view(b, num_tokens, self.num_heads, self.head_dim)
        value = value.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # transpose the dimension so that num_tokens and head_dim are 2 last layer
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_score = query @ key.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_score.masked_fill_(mask_bool, -float("inf"))
        attn_weight = torch.softmax(attn_score / self.head_dim ** 0.5, dim=-1)

        attn_weight = self.dropout(attn_weight)

        # reverse the transpose from previous
        context_vec = (attn_weight @ value).transpose(1, 2)
        # the contiguous is to be sure that the memory in a block for the view to execute
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # make the output of those head smoothout since it is just concat form those head
        context_vec = self.out_proj(context_vec)
        return context_vec