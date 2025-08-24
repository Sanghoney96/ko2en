import torch.nn as nn
from models.layers import *


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout=dropout, is_causal=False
        )
        self.ffn = FeedForwardNet(d_model, d_ff, dropout)

    def forward(self, x, padding_mask=None):
        # Self-attention block (Pre-LN)
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, padding_mask)
        x = x + attn_out

        # FeedForward block (Pre-LN)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout=dropout, is_causal=True
        )
        self.cross_attn = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout=dropout, is_causal=False
        )
        self.ffn = FeedForwardNet(d_model, d_ff, dropout)

    def forward(self, x, enc_out, tgt_padding_mask=None, src_padding_mask=None):
        # masked self-attention
        x_norm = self.norm1(x)
        self_attn_out = self.masked_self_attn(x_norm, x_norm, x_norm, tgt_padding_mask)
        x = x + self_attn_out

        # cross-attention (query: decoder, key/value: encoder)
        x_norm = self.norm2(x)
        cross_attn_out = self.cross_attn(x_norm, enc_out, enc_out, src_padding_mask)
        x = x + cross_attn_out

        # FFN
        x_norm = self.norm3(x)
        ff_out = self.ffn(x_norm)
        x = x + ff_out

        return x
