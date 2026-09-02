import tiktoken # byte-pair tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from llm import GPTModel, GPTDataset
import numpy as np

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

train_path = "fineweb_train.bin"
val_path = "fineweb_train.bin"

def text_to_token_ids(tokenizer, text):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded)
    if (len(encoded_tensor.shape) == 1):
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(tokenizer, idx):
    decoded = tokenizer.decode(idx.squeeze(0).tolist())
    return decoded
        
tokenizer = tiktoken.get_encoding("gpt2")
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

train_token = GPTDataset(
    "./data/fineweb_train.bin", 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"]
)

val_token = GPTDataset(
    "./data/fineweb_val.bin",
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"]
)


train_loader = DataLoader(
    train_token,
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = DataLoader(
    val_token,
    batch_size=2,
    shuffle=False,
    drop_last=False,
    num_workers=0
)



def cal_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten(0, 1)
        ) 
    return loss

def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    sum_loss = 0
    
    for i, (data_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch=data_batch, target_batch=target_batch, model=model, device=device)
            sum_loss += loss.item()
    return sum_loss / num_batches    

def generate(model, idx, max_new_tokens, context_size, temparature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens): 
        idx = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(-float("inf")).to(device),
                logits
            )
        
        if temparature > 0:
            logits = logits / temparature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx 

def generate_and_print_sample(model, tokenizer, device, start_context, temparature):
    model.eval()
    context_size=model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(tokenizer=tokenizer, text=start_context).to(device)

    with torch.no_grad():
        encoded = generate(model, encoded, 50, context_size=context_size, temparature=1.0, top_k=20) 
    
    decoded = token_ids_to_text(tokenizer=tokenizer, idx=encoded)
    print(decoded.replace("\n", " "))
    model.train()

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    token_seen,
    train_losses,
    val_losses,
    path
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        "epoch": epoch,
        "global_step": global_step,
        "token_seen": token_seen,

        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    torch.save(checkpoint, path)
    
def training_model(model, train_loader, val_loader, optimizer, 
                   device, num_epochs, eval_freq, 
                   eval_iter, start_context, tokenizer, scaler):
   
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(input_batch)

                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    target_batch.flatten(0, 1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): ")
                print(f"Train loss {train_loss: .3f}")
                print(f"Val loss {val_loss: .3f}")

    return train_losses, val_losses, track_token_seen 

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.to(device=device)
scaler = torch.amp.GradScaler("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)            

num_epochs = 30
train_losses, val_losses, tokens_seen = training_model(model=model, train_loader=train_loader, val_loader=val_loader,
                                                       optimizer=optimizer, tokenizer=tokenizer, num_epochs=num_epochs, eval_freq=500,
                                                       eval_iter=5, start_context="Hi, I am", device=device, scaler=scaler)

torch.save(model.state_dict(), "gpt124m_final.pt")
              

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)