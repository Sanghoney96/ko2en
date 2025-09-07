import torch
import torch.nn as nn


def loss_function(logits, targets):
    """
    logits: (batch_size, seq_len, vocab_size), targets: (batch_size, seq_len)
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    return loss_fn(logits.permute(0, 2, 1), targets)
