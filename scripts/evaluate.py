import os
from math import exp

import tiktoken
import torch
from paths import CHECKPOINT_DIR, DATA_DIR
from torch.utils.data import DataLoader
from training import calc_loss_loader, generate

from configs.model_config import GPT_CONFIG
from llm import GPTDataset, GPTModel


def load_model(checkpoint_path, device):
    model = GPTModel(GPT_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, device, max_tokens=50):
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    token_ids = generate(
        model,
        token_ids,
        max_new_tokens=max_tokens,
        context_size=GPT_CONFIG["context_length"],
        temparature=0.8,
        top_k=20,
    )
    return tokenizer.decode(token_ids.squeeze(0).tolist())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model(
        os.path.join(CHECKPOINT_DIR, "pretraining", "best.pth"),
        device,
    )

    val_dataset = GPTDataset(
        os.path.join(DATA_DIR, "writing_val.bin"),
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loss = calc_loss_loader(val_loader, model, device, 5000)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Perplexity: {exp(val_loss):.2f}")
    print(generate_text(model, tokenizer, "A student should", device))
