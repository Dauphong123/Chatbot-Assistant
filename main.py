import tiktoken # byte-pair tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super(GPTDataset, self).__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index] 
        
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


class CausalAttentionLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.1, qkv_bias=False):
        super(CausalAttentionLayer, self).__init__()
        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, input):
        query = self.W_Q(input)
        key = self.W_K(input)
        value = self.W_V(input)

        attn_score = torch.matmul(query, key.transpose(-2, -1))
        attn_score = self.dropout(attn_score.masked_fill(self.mask.bool()[:input.shape[-2], :input.shape[-2]], -float('inf')))
        attn_weight = torch.softmax(attn_score / input.shape[-1] ** 0.5, dim=-1)
        outputs = torch.matmul(attn_weight, value)
        return outputs
    
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
        attn_weight = torch.softmax(attn_score / num_tokens ** 0.5, dim=-1)

        attn_weight = self.dropout(attn_weight)

        # reverse the transpose from previous
        context_vec = (attn_weight @ value).transpose(1, 2)
        # the contiguous is to be sure that the memory in a block for the view to execute
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # make the output of those head smoothout since it is just concat form those head
        context_vec = self.out_proj(context_vec)
        return context_vec
    
class NormLayer(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        eps = 1e-8
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + eps)
        return self.scale * norm_x + self.shift



def create_loader(txt, batch_size=4, max_length=128, stride=128, shuffle=True, drop_last=True, num_worker=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt=raw_text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_worker
    )
    return dataloader

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_loader(raw_text, batch_size=1, max_length=4, stride = 4)

