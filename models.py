import numpy as np
import torch

def positional_encoding(input_len, d_model):
    pos = np.arange(input_len).reshape(-1, 1)  # (input_len, 1)
    i = np.arange(d_model).reshape(1, -1)  # (1, d_model)

    angle_rads = pos / np.power(10000, (2 * (i // 2)) / d_model)  # (m, d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads