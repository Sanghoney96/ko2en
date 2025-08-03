import torch
import torch.nn as nn
from layers import PositionalEncoding
from blocks import Encoder, Decoder
from utils import get_padding_mask


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        src_len,
        tgt_len,
        d_model,
        d_ff,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout=0.1,
    ):
        super().__init__()
        # 임베딩 & 포지셔널 인코딩
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        self.src_pos_encoding = PositionalEncoding(src_len, d_model)
        self.tgt_pos_encoding = PositionalEncoding(tgt_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # 인코더 스택
        self.encoder_layers = nn.ModuleList(
            [
                Encoder(d_model, d_ff, n_heads, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # 디코더 스택
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(d_model, d_ff, n_heads, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        # 출력 projection
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_input_ids, tgt_input_ids):
        # 인코더 입력 마스크
        src_padding_mask = get_padding_mask(src_input_ids, pad_token_id=0)
        tgt_padding_mask = get_padding_mask(tgt_input_ids, pad_token_id=0)

        # 인코더 입력 처리
        enc_x = self.src_embedding(src_input_ids)
        enc_x = self.src_pos_encoding(enc_x)
        enc_x = self.dropout(enc_x)

        for layer in self.encoder_layers:
            enc_x = layer(enc_x, src_padding_mask)

        # 디코더 입력 처리
        dec_x = self.tgt_embedding(tgt_input_ids)
        dec_x = self.tgt_pos_encoding(dec_x)
        dec_x = self.dropout(dec_x)

        for layer in self.decoder_layers:
            dec_x = layer(dec_x, enc_x, tgt_padding_mask, src_padding_mask)

        # 최종 출력
        logits = self.out_proj(dec_x)  # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    model = Transformer(
        vocab_size=10000,
        src_len=50,  # 인코더 입력 길이
        tgt_len=40,  # 디코더 입력 길이
        d_model=128,
        d_ff=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.3,
    )

    src_input_ids = torch.randint(0, 10000, (64, 50))
    tgt_input_ids = torch.randint(0, 10000, (64, 40))

    logits = model(src_input_ids, tgt_input_ids)
    print(logits.shape)  # (64, 40, 10000)
