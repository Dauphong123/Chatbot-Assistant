import os
import time

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch.utils.tensorboard import SummaryWriter

from scripts.paths import PLOTS_DIR


def text_to_token_ids(tokenizer, text):
    """
    Convert text to token ids

    Args:
        tokenizer (): tiktoken tokenizer
        text (string): the text

    Returns:
        encoded_tensor: torch tensor of the text
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded)
    if len(encoded_tensor.shape) == 1:
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(tokenizer, idx):
    """
    turn token ids to text

    Args:
        tokenizer (): tiktoken tokenizer
        idx (string): the ids

    Returns:
        string: the text from the ids
    """
    return tokenizer.decode(idx.squeeze(0).tolist())


def cal_loss_batch(input_batch, target_batch, model, device):
    """
    calculate the loss of the batch from the model

    Args:
        input_batch (torch.tensor):
            the input matrix of shape (batch_size, context_size)
        target_batch (torch.tensor):
            the target matrix of shape (batch_size, context_size)
        model (torch.nn.Module):
            model of the code to predict the target
        device (torch.device):
            device to run the code (CPU or GPU)

    Returns:
        loss: the loss of the batch using cross entropy
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten(0, 1),
        )
    return loss


def calc_loss_loader(loader, model, device, num_batches=None):
    """
    calculate the loss of the whole data loader

    Args:
        loader (torch.utils.data.DataLoader): data loader
        model (torch.nn.Module): model
        device (torch.device): device
        num_batches (int): number of batches to calculate loss. Defaults to None.

    Returns:
        return the loss of the loader
    """
    if len(loader) == 0:
        print(loader[0])
        return float("nan")
    if num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))

    sum_loss = 0.0
    valid_batches = 0
    for i, (data_batch, target_batch) in enumerate(loader):
        if i >= num_batches:
            break
        if not (target_batch != -100).any():
            continue
        loss = cal_loss_batch(data_batch, target_batch, model, device)
        sum_loss += loss.item()
        valid_batches += 1
    return sum_loss / valid_batches if valid_batches else float("nan")


def generate(
    model, idx, max_new_tokens, context_size, temparature=0.0, top_k=None, eos_id=None
):
    """
    generate the next tokens depend on the input tokens with the model

    Args:
        model (torch.nn.Module): current model
        idx (torch.tensor): inputs ids
        max_new_tokens (torch.float16): number of tokens need to generate
        context_size (torch.float16): length of context size
        temparature (float, optional): temperature of the generator generate sampling. Defaults to 0.0.
        top_k (_type_, optional): number of outputs that allow to be generated. Defaults to None.
        eos_id (_type_, optional): id of the end of sequence tokens. Defaults to None.

    Returns:
        idx(torch.tensor) return a sequence of tokens that is the inputs tokens with addition of the output tokens from model
    """
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
            idx = torch.cat((idx, idx_next), dim=1)
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, temparature):
    """
    using generate function to from inputs text print output text to terminalto terminal

    Args:
        model (torch.nn.Module): current model
        tokenizer (tiktoken.core.Encodding): current tokenizer to change text to ids and vice versa
        device (torch.device): the device that model run on
        start_context (string): the string that input
        temparature (int): the temparature
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(tokenizer, start_context).to(device)
    with torch.no_grad():
        encoded = generate(model, encoded, 50, context_size, 1.0, temparature, 20)
    print(token_ids_to_text(tokenizer, encoded).replace("\n", " "))
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Using the train loader and validation loader to evaluate model using train loss and val loss. Therefore identify bias and variance

    Args:
        model (torch.nn.Module): current model
        train_loader (torch.utils.data.DataLoader): loader of the train data
        val_loader (torch.utils.data.DataLoader): load of the validation data
        device (torch.device): device that model run on
        eval_iter (int): number of batches that use to calculate loss

    Returns:
        return training loss and validation loss from 2 loader
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    optimizer_step,
    token_seen,
):
    """
    save checkpoint

    Args:
        path (str): checkpoint path
        model (torch.nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        scaler (torch.amp.grad_scaler.GradScaler): scaler
        epoch (int): epoch num
        global_step (int): global step num
        optimizer_step (int): optimizer step num
        token_seen (int): token seen num
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "token_seen": token_seen,
        },
        path,
    )


def training_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    scaler,
    accumulate_step,
    checkpoint_dir,
    log_dir=None,
    checkpoint_path=None,
):
    """
    training the model

    Args:
        model (torch.nn.Module): initial model without training
        train_loader (torch.utils.data.DataLoader): data loader of training data
        val_loader (torch.utils.data.DataLoader): data loader of validation data
        optimizer (torch.optim.Optimizer): optimizer of the model
        schduler (torch.optim.scheduler_lr.Scheduler) lr scheduler for optimizer
        device (torch.device): device that model train on
        num_epochs (int): number of epochs
        eval_freq (int): number of step between each evaluation
        eval_iter (int): number of batches in each evaluation
        scaler (torch.amp.grad_scaler.GradScaler): scaler to change data to fp16 instead of fp32
        accumulate_step (int): number of step after each gradient update
        checkpoint_dir (string): directory of the checkpoint file
        warmup_steps (int): number of step till the full learning rate
        peak_steps (int): number of step after reaching the full learning rate till the scheduled drop
        min_lr (int): learning rate of optimizer after the scheduled drop
        log_dir (string): directory for TensorBoard logs
        checkpoint_path (string): optional checkpoint to resume from

    Returns:
        return all the train losses, val losses and number of tokens using to train
    """

    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step, optimizer_step = 0, 0, 0
    start_epoch = 0
    best_val_loss = float("inf")

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", -1) + 1
            global_step = checkpoint.get("global_step", 0)
            token_seen = checkpoint.get("token_seen", 0)
            optimizer_step = checkpoint.get("optimizer_step", 0)
        else:
            model.load_state_dict(checkpoint)

    last_eval_time = time.perf_counter()
    last_token_seen = 0
    writer = SummaryWriter(log_dir) if log_dir else None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for step, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            if not (target_batch != -100).any():
                continue
            ##  AMP reduces GPU memory usage and can accelerate computation by using lower precision where appropriate.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_batch)
                loss = (
                    torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1),
                        target_batch.flatten(0, 1),
                        ignore_index=-100,
                    )
                    / accumulate_step
                )

            # since fp16 is sometimes underflowing using GradScaler to prevent that
            scaler.scale(loss).backward()
            if (step + 1) % accumulate_step == 0:
                # unscale before clipping
                scaler.unscale_(optimizer)

                # grad clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                # update parameter
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                optimizer_step += 1

            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                now = time.perf_counter()
                elapsed = now - last_eval_time
                last_eval_time = now
                if writer is not None:
                    writer.add_scalars(
                        "loss",
                        {"train": train_loss, "Val": val_loss},
                        global_step,
                    )
                    writer.add_scalar(
                        "training/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                    writer.add_scalar(
                        "training_speed",
                        (token_seen - last_token_seen) / elapsed,
                        global_step,
                    )
                    last_token_seen = token_seen
                    writer.add_scalar("training/tokens_seen", token_seen, global_step)
                    writer.add_scalar("training/epoch", epoch + 1, global_step)
                    writer.flush()
                save_checkpoint(
                    os.path.join(checkpoint_dir, "latest.pth"),
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    optimizer_step,
                    token_seen,
                )

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        os.path.join(checkpoint_dir, "best.pth"),
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        global_step,
                        optimizer_step,
                        token_seen,
                    )

        print("this is the end of epoch")

    if writer is not None:
        writer.close()

    return train_losses, val_losses, track_token_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    function to plot the training losses and val losses arcording to tokens_seen

    Args:
        epochs_seen (int): number of epochs
        tokens_seen (int): tokens seen
        train_losses (int): train losses
        val_losses (int): validation losses
    """
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
