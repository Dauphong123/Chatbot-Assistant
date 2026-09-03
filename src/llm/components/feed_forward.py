import torch
from .gelu import GELU

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim * 4),
            GELU(),
            torch.nn.Linear(emb_dim * 4, emb_dim) 
        )
    
    def forward(self, x):
        return self.layers(x)