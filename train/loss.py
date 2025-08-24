import torch
import torch.nn as nn


def loss_function(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0):
    """
    logits: (batch_size, seq_len, vocab_size), targets: (batch_size, seq_len)
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")
    return loss_fn(logits.permute(0, 2, 1), targets)
