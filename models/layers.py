import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mask_fn import get_causality_mask


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)  # (max_len, 1)
        i = torch.arange(d_model, dtype=torch.float32).reshape(1, -1)  # (1, d_model)

        # Set an angle on each (pos, 2i)
        # shape : (input_len, d_model)
        angle_rads = pos / torch.pow(10000, (2 * (i // 2)) / d_model)

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        # Set angle_rads as non-trainable,
        # then ensure pos_encoding is on the same device as input tensor.
        pos_encoding = angle_rads.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len]

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout, is_causal=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.is_causal = is_causal

    def split_heads(self, x):
        """
        (batch_size, input_len, d_model) -> (batch_size, n_heads, input_len, d_k)
        """
        batch_size, input_len = x.shape[0], x.shape[1]
        x_split = x.reshape(batch_size, input_len, self.n_heads, self.d_k)
        x_split = x_split.permute(0, 2, 1, 3)

        return x_split

    def scaled_dot_product_attention(
        self, q, k, v, causality_mask=None, padding_mask=None
    ):
        k = k.transpose(2, 3)
        scores = torch.matmul(q, k) / math.sqrt(self.d_k)

        # Apply causality mask if provided (to prevent attending to future positions)
        if causality_mask is not None:
            scores = scores + causality_mask

        # Apply padding mask if provided (to prevent attending to padding tokens)
        if padding_mask is not None:
            scores = scores + padding_mask

        attn_map = F.softmax(scores, dim=-1)
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

    def forward(self, query, key, value, padding_mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # Generate causality mask if required
        if self.is_causal:
            causality_mask = get_causality_mask(seq_len)
            causality_mask = causality_mask.to(query.device).type_as(query)
            causality_mask = causality_mask.unsqueeze(0).unsqueeze(
                0
            )  # (1, 1, seq, seq)
        else:
            causality_mask = None

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # (batch_size, n_heads, input_len, d_k)
        heads = self.scaled_dot_product_attention(Q, K, V, causality_mask, padding_mask)

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
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        out = self.dropout_2(x)

        return out
