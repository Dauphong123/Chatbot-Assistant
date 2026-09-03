from torch.utils.data import DataLoader, Dataset
import torch 
from .components import NormLayer, MultiheadAttentionLayer, FeedForward

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.norm1 = NormLayer(cfg["emb_dim"])
        self.norm2 = NormLayer(cfg["emb_dim"])
        self.ff = FeedForward(cfg["emb_dim"])
        self.attn = MultiheadAttentionLayer(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=0.1,
            num_heads=cfg["n_heads"],   
            qkv_bias=cfg["qkv_bias"]
        )
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, input):
        skip = input # shortcut
        x = self.norm1(input)
        x = self.attn(x)
        x = self.dropout(x)
        x += skip

        skip = x # shortcut
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x += skip
        return x
    
class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) 
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = torch.nn.Dropout(cfg["drop_rate"])

        self.trf_block = torch.nn.Sequential(
            *[TransformerBlock(cfg) for i in range(cfg["n_layers"])]
        )

        self.final_norm = NormLayer(cfg["emb_dim"])
        self.out_head = torch.nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, input):
        # token emb -> transformerblock -> NormLayer -> output
        b, tok_length = input.shape
        tok_emb = self.tok_emb(input) 
        pos_emb = self.pos_emb(
            torch.arange(tok_length, device=input.device)
        )
        
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x) 
        return logits