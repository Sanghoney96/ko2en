import torch
from torch.amp import autocast, GradScaler

print(torch.__version__)


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, device):
    model.train()

    scaler = GradScaler()

    total_loss = 0.0
    steps = 0
    log_every = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        enc_in = batch["encoder_input_ids"].to(device)
        dec_in = batch["decoder_input_ids"].to(device)
        labels = batch["label_ids"].to(device)

        with autocast(device_type="cuda"):
            # calculate prediction and loss
            logits = model(enc_in, dec_in)
            logits = logits.transpose(1, 2).contiguous()
            loss = loss_fn(logits, labels)

        # 역전파
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # logging
        loss_item = loss.detach().item()
        total_loss += loss_item
        steps += 1
        if (batch_idx + 1) % log_every == 0:
            print(
                f"[train] step {batch_idx} | loss {loss_item:.4f} | avg {total_loss/steps:.4f}"
            )

    return total_loss / max(1, steps)
