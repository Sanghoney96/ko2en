import torch
import torch.nn.functional as F


def loss_function(logits, targets):
    """
    logits: (batch_size, seq_len, vocab_size), targets: (batch_size, seq_len)
    """
    loss_fn = F.cross_entropy(logits, targets, ignore_index=0, reduction="mean")
    return loss_fn
