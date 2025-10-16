import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Efficient sinusoidal positional encoding (absolute)."""
    def __init__(self, emb_dim: int, dropout: float = 0.1, seq_len: int = 8192):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even, got {emb_dim}")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(seq_len, emb_dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, seq_len, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings."""
    def __init__(self, emb_dim: int, dropout: float = 0.1, seq_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(seq_len, emb_dim)
        self.register_buffer('positions', torch.arange(seq_len, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos_embeddings = self.position_embedding(self.positions[:T])  # (T, D)
        x = x + pos_embeddings
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE). Apply to Q and K inside attention."""
    def __init__(self, emb_dim: int, seq_len: int = 8192, n: int = 10000):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"Embedding dim must be even for RoPE, got {emb_dim}")
        inv_freq = 1.0 / (n ** (torch.arange(0, emb_dim, 2, dtype=torch.float) / emb_dim))
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, D)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)  # (1,T,1,D)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)  # (1,T,1,D)

    def _apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, Dh) Dh must be even
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

        T = x.shape[1]
        cos = self.cos_cached[:, :T, :, :]  # (1,T,1,D)
        sin = self.sin_cached[:, :T, :, :]
        out = x * cos + rotated * sin
        return out.type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self._apply_rotary_emb(q), self._apply_rotary_emb(k)


class ALiBi(nn.Module):
    """
    ALiBi: attention with linear biases. Add to attention scores (before softmax).
    """
    def __init__(self, num_heads: int, seq_len: int = 8192):
        super().__init__()
        self.num_heads = num_heads
        slopes = torch.tensor(self._get_slopes(num_heads), dtype=torch.float)  # (H,)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # (1,T)
        distances = (pos.transpose(0, 1) - pos).abs()  # (T,T)
        alibi_bias = distances.unsqueeze(0) * slopes.view(-1, 1, 1)  # (H,T,T)
        self.register_buffer("alibi_bias", -alibi_bias, persistent=False)

    @staticmethod
    def _get_slopes(n: int):
        # From ALiBi paper reference implementation
        import math
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        # attn_scores: (B,H,T,T)
        T = attn_scores.size(-1)
        bias = self.alibi_bias[:, :T, :T].unsqueeze(0)  # (1,H,T,T)
        return attn_scores + bias