import math
import os
from functools import partial

import tiktoken  # byte-pair tokenizer
import torch
from datasets import load_dataset
from paths import CHECKPOINT_DIR, DATA_DIR, ROOT_DIR
from torch.utils.data import DataLoader
from training import plot_losses, training_model

from configs.model_config import FINETUNING_CONFIG, GPT_CONFIG
from llm import GPTModel

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TRAIN_PATH = os.path.join(DATA_DIR, "SFT_train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "SFT_val.parquet")
PRETRAINING_PATH = os.path.join(CHECKPOINT_DIR, "pretraining", "best.pth")
IGNORE_INDEX = -100
NUM_EPOCH = 2
ACCUMULATE_STEP = 2

tokenizer = tiktoken.get_encoding("gpt2")


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
):
    if allowed_max_length is not None:
        batch_max_length = min(
            max(len(item["input_ids"]) for item in batch),
            allowed_max_length,
        )
    else:
        batch_max_length = max(len(item["input_ids"]) for item in batch)

    input_lst = []
    target_lst = []

    for item in batch:
        input_ids = item["input_ids"][:batch_max_length]
        labels = item["labels"][:batch_max_length]

        padding_length = batch_max_length - len(input_ids)

        input_ids = input_ids + [pad_token_id] * padding_length
        labels = labels + [ignore_index] * padding_length

        input_lst.append(torch.tensor(input_ids, dtype=torch.long))
        target_lst.append(torch.tensor(labels, dtype=torch.long))

    return (
        torch.stack(input_lst),
        torch.stack(target_lst),
    )


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_files = {"train": TRAIN_PATH, "val": VAL_PATH}

    dataset = load_dataset("parquet", data_files=data_files)

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    print("dataset loaded")

    customized_collate_fn = partial(
        custom_collate_fn, allowed_max_length=GPT_CONFIG["context_length"]
    )

    train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        collate_fn=customized_collate_fn,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,  # type: ignore
        batch_size=2,
        shuffle=False,
        collate_fn=customized_collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    model = GPTModel(cfg=GPT_CONFIG)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FINETUNING_CONFIG["learning_rate"], weight_decay=0.01
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=FINETUNING_CONFIG["warmup_steps"],
    )

    total_optimizer_step = math.ceil(len(train_loader) / ACCUMULATE_STEP) * NUM_EPOCH
    cosine_steps = total_optimizer_step - FINETUNING_CONFIG["warmup_steps"]
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=FINETUNING_CONFIG["min_learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[
            FINETUNING_CONFIG["warmup_steps"],
        ],
    )
    scaler = torch.GradScaler(device.type, enabled=device.type == "cuda")
    checkpoint = torch.load(PRETRAINING_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    os.makedirs(os.path.join(CHECKPOINT_DIR, "finetuning"), exist_ok=True)
    train_losses, val_losses, tokens_seen = training_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        NUM_EPOCH,
        eval_freq=1000,
        eval_iter=100,
        scaler=scaler,
        accumulate_step=ACCUMULATE_STEP,
        checkpoint_dir=os.path.join(CHECKPOINT_DIR, "finetuning"),
        log_dir=os.path.join(ROOT_DIR, "runs/gpt_finetuning"),
    )

    epochs_tensor = torch.linspace(0, NUM_EPOCH, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
