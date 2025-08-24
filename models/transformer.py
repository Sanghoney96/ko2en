import math
import torch
import torch.nn as nn
from models.layers import PositionalEncoding
from models.blocks import Encoder, Decoder
from utils import get_padding_mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
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

        self.d_model = d_model

        # 임베딩 & 포지셔널 인코딩
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.src_pos_encoding = PositionalEncoding(src_len, d_model)
        self.tgt_pos_encoding = PositionalEncoding(tgt_len, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

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
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_input_ids, tgt_input_ids):
        # 인코더 입력 마스크
        src_padding_mask = get_padding_mask(src_input_ids, pad_token_id=0)
        tgt_padding_mask = get_padding_mask(tgt_input_ids, pad_token_id=0)

        # 인코더 입력 처리
        enc_x = self.src_embedding(src_input_ids) * math.sqrt(self.d_model)
        enc_x = self.src_pos_encoding(enc_x)
        enc_x = self.dropout1(enc_x)

        for layer in self.encoder_layers:
            enc_x = layer(enc_x, src_padding_mask)

        enc_x = self.norm1(enc_x)

        # 디코더 입력 처리
        dec_x = self.tgt_embedding(tgt_input_ids) * math.sqrt(self.d_model)
        dec_x = self.tgt_pos_encoding(dec_x)
        dec_x = self.dropout2(dec_x)

        for layer in self.decoder_layers:
            dec_x = layer(dec_x, enc_x, tgt_padding_mask, src_padding_mask)

        dec_x = self.norm2(dec_x)

        # 최종 출력
        logits = self.out_proj(dec_x)  # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성 및 디바이스 이동
    model = Transformer(
        src_vocab_size=30000,
        tgt_vocab_size=27000,
        src_len=55,  # 인코더 입력 길이
        tgt_len=55,  # 디코더 입력 길이
        d_model=128,
        d_ff=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.3,
    ).to(device)

    # 입력 텐서도 동일한 디바이스로 이동
    src_input_ids = torch.randint(0, 30000, (64, 55)).to(device)
    tgt_input_ids = torch.randint(0, 27000, (64, 55)).to(device)

    # 모델 실행
    logits = model(src_input_ids, tgt_input_ids)
    print(logits.shape)  # (64, 55, 27000)
