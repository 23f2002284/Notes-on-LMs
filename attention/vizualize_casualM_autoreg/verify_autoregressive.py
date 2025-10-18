"""
Verification of causal masking and autoregressive property.

This script:
  1) Verifies attention weights are zero above the diagonal (no look-ahead).
  2) Verifies outputs up to position t are invariant to changes in future tokens (> t).
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from attention_masks import make_causal_mask

# Minimal MHA implementation compatible with mask: 1=keep, 0=mask-out
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,T,d_k]

    def forward(self, q, k, v, mask=None, return_attn=False):
        B, T, _ = q.shape
        S = k.shape[1]
        q = self._split_heads(self.W_q(q))
        k = self._split_heads(self.W_k(k))
        v = self._split_heads(self.W_v(v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,T,S]

        if mask is not None:
            # Assume mask is 1.0 for keep, 0.0 for mask-out
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B,H,T,S]
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,T,d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)
        if return_attn:
            return out, attn
        return out

def verify_no_lookahead(attn, atol=1e-7):
    """
    attn: [B,H,T,S]
    Checks that for all i, j>i -> attn[..., i, j] == 0
    """
    B, H, T, S = attn.shape
    # Build an upper-triangular mask of "future" positions j>i
    future_mask = torch.triu(torch.ones((T, S), dtype=torch.bool, device=attn.device), diagonal=1)  # [T,S]
    max_future = attn[:, :, future_mask].abs().max().item()
    return max_future <= atol, max_future

def verify_output_invariance(model, x, mask, num_checks=3, dropout_off=True, atol=1e-6):
    """
    For several t, change future tokens (>t) and check outputs <= t stay identical.
    """
    if dropout_off:
        model.eval()
    with torch.no_grad():
        out_ref, _ = model(x, x, x, mask=mask, return_attn=True)

    B, T, D = x.shape
    rng = torch.Generator(device=x.device)
    rng.manual_seed(1234)
    ts = torch.linspace(0, T - 2, steps=min(num_checks, T - 1)).round().to(torch.long).tolist()
    for t in ts:
        x2 = x.clone()
        # Replace future tokens (>t) with new random values
        x2[:, t + 1 :, :] = torch.randn_like(x2[:, t + 1 :, :], generator=rng)
        with torch.no_grad():
            out2, _ = model(x2, x2, x2, mask=mask, return_attn=True)
        # Compare prefixes up to t (inclusive)
        diff = (out_ref[:, : t + 1, :] - out2[:, : t + 1, :]).abs().max().item()
        if diff > atol:
            return False, t, diff
    return True, None, 0.0

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    B, T, D, H = 2, 8, 32, 4
    x = torch.randn(B, T, D, device=device)
    model = MultiHeadAttention(d_model=D, num_heads=H, dropout=0.0).to(device)

    mask = make_causal_mask(T=T, S=T, batch_size=B, num_heads=H, device=device, dtype=torch.float32)

    # Forward with mask
    out, attn = model(x, x, x, mask=mask, return_attn=True)

    # 1) No-lookahead check
    ok1, max_future = verify_no_lookahead(attn, atol=1e-7)
    print(f"No-lookahead (future attention == 0): {ok1}, max future weight = {max_future:.2e}")

    # 2) Output invariance to future changes
    ok2, t_bad, diff = verify_output_invariance(model, x, mask, num_checks=3, atol=1e-6)
    if ok2:
        print("Output invariance to future changes: True (all checked prefixes unchanged)")
    else:
        print(f"Output invariance to future changes: False (prefix up to t={t_bad} differs by {diff:.3e})")