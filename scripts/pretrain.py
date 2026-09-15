import os

import torch
from paths import CHECKPOINT_DIR, DATA_DIR, LOG_DIR
from torch.utils.data import DataLoader
from training import plot_losses, training_model

from configs.model_config import GPT_CONFIG, PRETRAINING_CONFIG
from llm import GPTDataset, GPTModel

ACCUMULATE_STEP = 2
NUM_EPOCH = 1

TRAIN_PATH = os.path.join(DATA_DIR, "writing_train.bin")
VAL_PATH = os.path.join(DATA_DIR, "writing_val.bin")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to a checkpoint to resume training from",
    )
    args = parser.parse_args()
    """

    train_token = GPTDataset(
        TRAIN_PATH,
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
    )

    val_token = GPTDataset(
        VAL_PATH,
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
    )

    train_loader = DataLoader(
        train_token,
        batch_size=2,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_token,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=2,
    )

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG)
    model.to(device=device)

    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PRETRAINING_CONFIG["learning_rate"],
        weight_decay=0.01,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=PRETRAINING_CONFIG["warmup_steps"],
    )

    total_optimizer_step = len(train_loader) // ACCUMULATE_STEP * NUM_EPOCH
    cosine_steps = total_optimizer_step - PRETRAINING_CONFIG["warmup_steps"]
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=PRETRAINING_CONFIG["min_learningrate"]
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[
            PRETRAINING_CONFIG["warmup_steps"],
        ],
    )

    print("Train samples:", len(train_token))
    print("Batches per epoch:", len(train_loader))
    print("Train file:", TRAIN_PATH)
    print("Train file bytes:", os.path.getsize(TRAIN_PATH))
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "pretraining", "best.pth")

    os.makedirs(os.path.join(CHECKPOINT_DIR, "pretraining"), exist_ok=True)
    train_losses, val_losses, tokens_seen = training_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        num_epochs=NUM_EPOCH,
        eval_freq=5000,
        eval_iter=50,
        device=device,
        scaler=scaler,
        accumulate_step=ACCUMULATE_STEP,
        checkpoint_dir=os.path.join(CHECKPOINT_DIR, "pretraining"),
        log_dir=os.path.join(LOG_DIR, "gpt_training"),
        checkpoint_path=checkpoint_path,
    )

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "gpt124m_final.pt"))

    epochs_tensor = torch.linspace(0, NUM_EPOCH, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
