import math
import torch
from torch.amp import autocast, GradScaler
import tqdm


def train_loop(
    dataloader, model, loss_fn, optimizer, scheduler, device, log_every=1000
):
    model.train()

    total_loss = 0.0
    steps = 0
    losses = []

    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        enc_in = batch["encoder_input_ids"].to(device, non_blocking=True)
        dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["label_ids"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # calculate prediction and loss
        logits = model(enc_in, dec_in)
        logits = logits.transpose(1, 2).contiguous()
        loss = loss_fn(logits, labels)

        # 역전파
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # logging
        loss_item = loss.item()
        losses.append(loss_item)
        total_loss += loss_item
        steps += 1

        if (batch_idx + 1) % log_every == 0:
            print(
                f"[train] step {batch_idx+1}/{len(dataloader)} | loss {loss_item:.4f}"
            )

    return total_loss / steps, losses


def eval_loop(dataloader, model, loss_fn, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            enc_in = batch["encoder_input_ids"].to(device, non_blocking=True)
            dec_in = batch["decoder_input_ids"].to(device, non_blocking=True)
            labels = batch["label_ids"].to(device, non_blocking=True)

            # calculate prediction and loss
            logits = model(enc_in, dec_in)
            logits = logits.transpose(1, 2).contiguous()
            loss = loss_fn(logits, labels)

            # logging
            loss_item = loss.item()
            total_loss += loss_item

    avg_eval_loss = total_loss / len(dataloader)
    ppl = math.exp(avg_eval_loss)

    return avg_eval_loss, ppl
