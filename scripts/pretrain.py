from paths import DATA_DIR, CHECKPOINT_DIR

import tiktoken # byte-pair tokenizer
import torch
from torch.utils.data import DataLoader
from llm import GPTModel, GPTDataset
import os
from configs.model_config import GPT_CONFIG_124M
from training import plot_losses, training_model

TRAIN_PATH = os.path.join(DATA_DIR, "writing_train.bin")
VAL_PATH = os.path.join(DATA_DIR, "writing_val.bin")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

tokenizer = tiktoken.get_encoding("gpt2")

train_token = GPTDataset(
    TRAIN_PATH, 
    max_length=GPT_CONFIG_124M["context_length"], 
    stride=GPT_CONFIG_124M["context_length"]
)

val_token = GPTDataset(
    VAL_PATH,
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

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.to(device=device)
scaler = torch.amp.GradScaler("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)            

num_epochs = 10
train_losses, val_losses, tokens_seen = training_model(model=model, train_loader=train_loader, val_loader=val_loader,
                                                       optimizer=optimizer, num_epochs=num_epochs, eval_freq=1000,
                                                       eval_iter=20, device=device, scaler=scaler, accumulate_step=8,
                                                       checkpoint_dir=CHECKPOINT_DIR)

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "gpt124m_final.pt"))
              

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)