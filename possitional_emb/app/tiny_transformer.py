import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    ALiBi,
)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        rope: Optional[RotaryPositionalEmbedding] = None,
        alibi: Optional[ALiBi] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = rope
        self.alibi = alibi

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B,T,D)
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        if self.rope is not None:
            # RoPE expects (B,T,H,Dh); we currently have (B,H,T,Dh) -> transpose
            q = q.transpose(1, 2)  # (B,T,H,Dh)
            k = k.transpose(1, 2)
            q, k = self.rope(q, k)
            q = q.transpose(1, 2)  # back to (B,H,T,Dh)
            k = k.transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        # causal mask: (1,1,T,T) with True where allowed; mask out others
        att = att.masked_fill(~causal_mask, float("-inf"))

        if self.alibi is not None:
            att = self.alibi(att)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.matmul(att, v)  # (B,H,T,Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B,T,D)
        y = self.proj_drop(self.proj(y))
        return y, (att if return_attn else None)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        rope: Optional[RotaryPositionalEmbedding] = None,
        alibi: Optional[ALiBi] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, rope, alibi)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, return_attn=False):
        attn_out, attn_weights = self.attn(self.ln1(x), causal_mask, return_attn=return_attn)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class TinyCausalTransformer(nn.Module):
    """
    A tiny decoder-only Transformer with swappable positional encoding strategies.
    positional_type in {"sinusoidal","learned","rope","alibi"}.
    For "rope": use rotary in attention.
    For "alibi": add linear bias to attention scores.
    For "sinusoidal"/"learned": add embedding to token embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        positional_type: str = "sinusoidal",
    ):
        super().__init__()
        assert positional_type in {"sinusoidal", "learned", "rope", "alibi"}
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.positional_type = positional_type

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.abs_pos = None
        self.rope = None
        self.alibi = None

        if positional_type == "sinusoidal":
            self.abs_pos = SinusoidalPositionalEncoding(d_model, dropout, seq_len=max_seq_len)
        elif positional_type == "learned":
            self.abs_pos = LearnedPositionalEmbedding(d_model, dropout, seq_len=max_seq_len)
        elif positional_type == "rope":
            self.rope = RotaryPositionalEmbedding(emb_dim=d_model // n_heads, seq_len=max_seq_len)
        elif positional_type == "alibi":
            self.alibi = ALiBi(num_heads=n_heads, seq_len=max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                rope=self.rope,
                alibi=self.alibi,
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Heads are task-specific; provided externally for LM or classification

        # causal mask buffer (max size); will slice to T
        cm = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        cm = cm.view(1, 1, max_seq_len, max_seq_len)  # (1,1,T,T)
        self.register_buffer("causal_mask_full", cm, persistent=False)

    def forward(self, idx: torch.Tensor, return_attn=False):
        # idx: (B,T)
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > model max_seq_len {self.max_seq_len}"

        x = self.tok_emb(idx)  # (B,T,D)
        if self.abs_pos is not None:
            x = self.abs_pos(x)
        else:
            x = self.drop(x)

        causal_mask = self.causal_mask_full[:, :, :T, :T]  # (1,1,T,T)
        attn_weights_all: List[torch.Tensor] = []
        for blk in self.blocks:
            x, attn_w = blk(x, causal_mask, return_attn=return_attn)
            if return_attn and attn_w is not None:
                attn_weights_all.append(attn_w.detach().cpu())

        x = self.ln_f(x)  # (B,T,D)
        return x, attn_weights_all