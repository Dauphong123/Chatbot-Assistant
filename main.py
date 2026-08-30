import tiktoken # byte-pair tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from llm import GPTModel, GPTDataset

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

def text_to_token_ids(tokenizer, text):
    encoded = tokenizer.encode(text, allowed_special={'|endoftext|'})
    encoded_tensor = torch.tensor(encoded)
    if (len(encoded_tensor.shape) == 1):
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(tokenizer, idx):
    decoded = tokenizer.decode(idx.squeeze(0).tolist())
    return decoded
        
tokenizer = tiktoken.get_encoding("gpt2")
with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

model = GPTModel(GPT_CONFIG_124M)

def simple_text_generate(idx, model, generate_num, context_length):
    for _ in range(generate_num):
        x = idx[: , -context_length:]
        with torch.no_grad():
            logits = model(x)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

train_ratio = 0.9
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

train_token = GPTDataset(
    train_data, 
    tokenizer=tokenizer, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"]
)

val_token = GPTDataset(
    val_data, 
    tokenizer=tokenizer, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"]
)

val_loader = DataLoader(
    train_token,
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = DataLoader(
    val_token,
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

