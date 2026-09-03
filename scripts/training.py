import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from paths import PLOTS_DIR

def text_to_token_ids(tokenizer, text):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded)
    if len(encoded_tensor.shape) == 1:
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(tokenizer, idx):
    return tokenizer.decode(idx.squeeze(0).tolist())


def cal_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten(0, 1),
        )
    return loss


def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    sum_loss = 0
    for i, (data_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            loss = cal_loss_batch(data_batch, target_batch, model, device)
            sum_loss += loss.item()
    return sum_loss / num_batches


def generate(model, idx, max_new_tokens, context_size, temparature=0.0,
             top_k=None, eos_id=None):
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
                torch.tensor(-float("inf")).to(logits.device),
                logits,
            )

        if temparature > 0:
            logits = logits / temparature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context,
                              temparature):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(tokenizer, start_context).to(device)
    with torch.no_grad():
        encoded = generate(model, encoded, 50, context_size, 1.0, top_k=20)
    print(token_ids_to_text(tokenizer, encoded).replace("\n", " "))
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def training_model(model, train_loader, val_loader, optimizer, device,
                   num_epochs, eval_freq, eval_iter, scaler,
                   accumulate_step, checkpoint_dir):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), target_batch.flatten(0, 1)
                ) / accumulate_step

            scaler.scale(loss).backward()
            if (step + 1) % accumulate_step == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            token_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                        },
                        os.path.join(checkpoint_dir, "best_model.pth"),
                    )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}):")
                print(f"Train loss {train_loss: .3f}")
                print(f"Val loss {val_loss: .3f}")
                print(f"Token {token_seen}")

    return train_losses, val_losses, track_token_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "training_loss.png"), dpi=150)
    plt.show()