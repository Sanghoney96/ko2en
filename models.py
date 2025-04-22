import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, input_len, d_model):
        super().__init__()

        pos = torch.arange(input_len).reshape(-1, 1)  # (input_len, 1)
        i = torch.arange(d_model).reshape(1, -1)  # (1, d_model)

        # Set an angle on each (pos, 2i)
        # shape : (input_len, d_model)
        angle_rads = pos / torch.pow(10000, (2 * (i // 2)) / d_model)

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        # Set angle_rads as non-trainable,
        # then ensure pos_encoding is on the same device as input tensor.
        self.register_buffer("pos_encoding", angle_rads)
        self.pos_encoding = self.pos_encoding.to(device)

    def forward(self, x):
        x = x + self.pos_encoding.unsqueeze(0)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        (batch_size, input_len, d_model) -> (batch_size, n_heads, input_len, d_k)
        """
        batch_size, input_len = x.shape[0], x.shape[1]
        x_split = x.reshape(batch_size, input_len, self.n_heads, self.d_k)
        x_split = x_split.permute(0, 2, 1, 3)

        return x_split

    def scaled_dot_product_attention(self, q, k, v):
        k = k.transpose(2, 3)
        scores = torch.matmul(q, k) / math.sqrt(self.d_k)
        attn_map = self.softmax(scores)
        out = torch.matmul(attn_map, v)

        return out

    def concat_heads(self, x):
        """
        (batch_size, n_heads, input_len, d_k) -> (batch_size, input_len, d_model)
        """
        batch_size, n_heads, input_len, d_k = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch_size, input_len, n_heads, d_k)
        x = x.reshape(batch_size, input_len, n_heads * d_k)
        return x

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # (batch_size, n_heads, input_len, d_k)
        heads = self.scaled_dot_product_attention(Q, K, V)

        concat = self.concat_heads(heads)  # (batch_size, input_len, d_model)
        out = self.W_o(concat)  # (batch_size, input_len, d_model)
        out = self.dropout(out)

        return out


class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        out = self.dropout(x)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    input_len = 40
    d_model = 128
    d_ff = 512
    n_heads = 8
    dropout = 0.5

    x = torch.rand((batch_size, input_len, d_model)).to(device)

    # Positional Encoding
    pos_encoding = PositionalEncoding(input_len, d_model).to(device)
    x = pos_encoding(x)
    print("After Positional Encoding:", x.shape)

    # Multi-Head Attention
    mha = MultiHeadAttention(n_heads, d_model, dropout).to(device)
    attn_out = mha(x)
    print("After Multi-Head Attention:", attn_out.shape)

    # 🔹 Residual + LayerNorm
    norm1 = nn.LayerNorm(d_model).to(device)
    x = norm1(x + attn_out)
    print("After Residual + LayerNorm (MHA):", x.shape)

    # FeedForward Network
    ffn = FeedForwardNet(d_model, d_ff, dropout).to(device)
    ff_out = ffn(x)
    print("After FeedForward Network:", ff_out.shape)

    # 🔹 Residual + LayerNorm
    norm2 = nn.LayerNorm(d_model).to(device)
    x = norm2(x + ff_out)
    print("After Residual + LayerNorm (FFN):", x.shape)
