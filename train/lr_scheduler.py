import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt


class NoamLR(_LRScheduler):
    """
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """

    def __init__(
        self,
        optimizer,
        d_model,
        warmup_steps=4000,
        last_epoch=-1,
    ):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # 1부터 시작
        scale = (self.d_model**-0.5) * min(step**-0.5, step * (self.warmup**-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]
