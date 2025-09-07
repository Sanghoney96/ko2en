import torch


def get_causality_mask(input_len):
    q_grid, k_grid = torch.meshgrid(torch.arange(input_len), torch.arange(input_len))
    causality_mask = torch.where(
        q_grid < k_grid, float("-inf"), torch.zeros_like(q_grid, dtype=torch.float32)
    )
    return causality_mask


def get_padding_mask(input, pad_token_id=0):
    # (batch_size, 1, 1, input_len)
    mask = (input == pad_token_id).unsqueeze(1).unsqueeze(2)
    padding_mask = mask.float() * float("-inf")
    return padding_mask
