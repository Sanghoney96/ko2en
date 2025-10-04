import torch


def get_causality_mask(input_len, device):
    mask = torch.ones((input_len, input_len), dtype=torch.bool).to(device)
    mask = torch.tril(mask).unsqueeze(0).unsqueeze(0)
    return mask


def get_padding_mask(input, pad_token_id=0):
    # (batch_size, 1, 1, input_len)
    mask = (input == pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask.to(torch.bfloat16) * -1e4


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     input_len = 50
#     mask = get_causality_mask(input_len, device)

#     print(mask)
