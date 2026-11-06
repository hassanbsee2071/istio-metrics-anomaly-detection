"""
model.py

The global forecasting model: ONE shared transformer across ALL services, with
a learned per-service embedding letting it specialize predictions per service
while sharing general traffic patterns across all of them.

Now predicts 4 real metrics (rate, error5xx, error4xx, p99) instead of 3.
"""

import torch
import torch.nn as nn

from config import N_FEATURES, N_REAL_FEATURES


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, emb_dim, drop_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), GELU(), nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg["emb_dim"], cfg["drop_rate"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.att(self.norm1(x))
        x = self.drop_shortcut(x) + shortcut

        shortcut = x
        x = self.ff(self.norm2(x))
        x = self.drop_shortcut(x) + shortcut
        return x


class MetricsTransformer(nn.Module):
    """Input:  (batch, window_size, N_FEATURES) ->
               [rate, error5xx, error4xx, p99, hour_sin, hour_cos], normalized
               + service_id: (batch,)
    Output: (batch, N_REAL_FEATURES) -> predicted next-step [rate, error5xx, error4xx, p99]
    """
    def __init__(self, num_services, n_features=N_FEATURES, n_out=N_REAL_FEATURES,
                 service_emb_dim=16, emb_dim=32, n_layers=2, n_heads=2,
                 context_length=60, drop_rate=0.1):
        super().__init__()
        self.service_emb = nn.Embedding(num_services, service_emb_dim)
        self.input_proj = nn.Linear(n_features + service_emb_dim, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        cfg = {"emb_dim": emb_dim, "n_heads": n_heads, "context_length": context_length,
               "drop_rate": drop_rate, "qkv_bias": False}
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(n_layers)])
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, n_out)

    def forward(self, x, service_id):
        b, seq_len, _ = x.shape
        svc = self.service_emb(service_id).unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([x, svc], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_emb(torch.arange(seq_len, device=x.device))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x[:, -1, :]
