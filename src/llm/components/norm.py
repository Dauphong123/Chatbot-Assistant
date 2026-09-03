import torch

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