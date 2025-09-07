import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # tensor로 변환
    src_list = [torch.as_tensor(b["src_ids"][1:-1], dtype=torch.long) for b in batch]
    tgt_list = [torch.as_tensor(b["tgt_ids"], dtype=torch.long) for b in batch]

    dec_in_list = [t[:-1] for t in tgt_list]  # [BOS ...]
    labels_list = [t[1:] for t in tgt_list]  # [... EOS]

    # padding
    src = pad_sequence(src_list, batch_first=True, padding_value=0)
    dec_in = pad_sequence(dec_in_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=0)

    return {"encoder_input_ids": src, "decoder_input_ids": dec_in, "label_ids": labels}
