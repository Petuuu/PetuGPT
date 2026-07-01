import config as C
import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_in=C.EMB_DIM, d_out=C.EMB_DIM, qkv_bias=C.QKV_BIAS):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        print("Self attention layer initializer")

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


class CausalAttentionLayer(nn.Module):
    def __init__(
        self,
        d_in=C.EMB_DIM,
        d_out=C.EMB_DIM,
        context_len=C.CONTEXT_LEN,
        dropout=C.DROPOUT,
        qkv_bias=C.QKV_BIAS,
    ):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

        print("Causal attention layer initializer")

    def forward(self, x):
        _, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        d_in=C.EMB_DIM,
        d_out=C.EMB_DIM,
        context_len=C.CONTEXT_LEN,
        dropout=C.DROPOUT,
        num_heads=C.N_HEADS,
        qkv_bias=C.QKV_BIAS,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisble by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = (
            self.W_query(x)
            .view(b, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys = (
            self.W_key(x)
            .view(b, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        values = (
            self.W_value(x)
            .view(b, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (
            (attn_weights @ values)
            .transpose(1, 2)
            .contiguous()
            .view(b, num_tokens, self.d_out)
        )
        return self.out_proj(context_vec)


if __name__ == "__main__":
    ca = MultiHeadAttentionLayer(2, 6, 4, 0.0, 1)
    cvs = ca(torch.rand(5, 3, 2))
    print(cvs)
    print(cvs.shape)
