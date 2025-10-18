from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Optional: set a nicer default style
plt.style.use("ggplot")


def plot_attention_heads(
    attn: torch.Tensor,
    batch_index: int = 0,
    tokens: Optional[List[str]] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
):
    """
    Plot per-head attention heatmaps for a single batch element.

    Args:
      attn: [B, H, T, S] attention weights
      batch_index: which batch item to visualize
      tokens: optional token strings for axis tick labels (length T and S if self-attn)
      title: optional figure title
      cmap: matplotlib colormap name
    """
    assert attn.dim() == 4, "attn must be [B, H, T, S]"
    B, H, T, S = attn.shape
    assert 0 <= batch_index < B, f"batch_index must be in [0,{B-1}]"

    attn_b = attn[batch_index].detach().cpu().numpy()  # [H, T, S]

    # Make a roughly square grid
    ncols = int(np.ceil(np.sqrt(H)))
    nrows = int(np.ceil(H / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), squeeze=False)
    fig.suptitle(title or f"Attention Heads (batch={batch_index})", fontsize=14)

    for h in range(H):
        r = h // ncols
        c = h % ncols
        ax = axes[r][c]
        im = ax.imshow(attn_b[h], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(f"Head {h}", fontsize=10)
        ax.set_xlabel("Keys (S)")
        ax.set_ylabel("Queries (T)")

        if tokens is not None:
            # Set ticks sparsely if many tokens
            max_ticks = 20
            xticks = np.linspace(0, S - 1, num=min(S, max_ticks), dtype=int)
            yticks = np.linspace(0, T - 1, num=min(T, max_ticks), dtype=int)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.grid(False)
            # Truncate token labels for readability
            fmt = lambda s: (s if len(s) <= 12 else s[:11] + "…")
            ax.set_xticklabels([fmt(tokens[i]) if i < len(tokens) else str(i) for i in xticks], rotation=90, fontsize=8)
            ax.set_yticklabels([fmt(tokens[i]) if i < len(tokens) else str(i) for i in yticks], fontsize=8)

        # Add a small colorbar per head
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Hide any unused subplots
    for h in range(H, nrows * ncols):
        r = h // ncols
        c = h % ncols
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/attn_heads.png", dpi=150)


def plot_attention_for_position(
    attn: torch.Tensor,
    position: int,
    batch_index: int = 0,
    tokens: Optional[List[str]] = None,
    cmap: str = "magma",
):
    """
    Plot per-head attention distribution for a single query position across keys.

    Args:
      attn: [B, H, T, S]
      position: query index (0..T-1)
      batch_index: which batch item to visualize
      tokens: optional token strings for x-axis labels (keys)
    """
    assert attn.dim() == 4, "attn must be [B, H, T, S]"
    B, H, T, S = attn.shape
    assert 0 <= batch_index < B
    assert 0 <= position < T

    attn_b = attn[batch_index]  # [H, T, S]
    fig, axes = plt.subplots(H, 1, figsize=(10, 2.0 * H), squeeze=False)
    fig.suptitle(f"Per-Head Attention for Query Position {position}", fontsize=14)

    for h in range(H):
        ax = axes[h][0]
        vals = attn_b[h, position].detach().cpu().numpy()  # [S]
        im = ax.imshow(vals[None, :], aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_yticks([])
        ax.set_ylabel(f"Head {h}")
        ax.set_xlabel("Keys (S)")
        ax.grid(False)
        if tokens is not None:
            max_ticks = 40
            xticks = np.linspace(0, S - 1, num=min(S, max_ticks), dtype=int)
            ax.set_xticks(xticks)
            fmt = lambda s: (s if len(s) <= 14 else s[:13] + "…")
            ax.set_xticklabels([fmt(tokens[i]) if i < len(tokens) else str(i) for i in xticks], rotation=90, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/attn_for_position.png", dpi=150)
    

if __name__ == "__main__":
    # Minimal demo with random tensors
    from MultiHeadAttnViz import MultiHeadAttentionViz

    torch.manual_seed(0)
    B, T, d_model, H = 1, 8, 32, 4
    x = torch.randn(B, T, d_model)

    # Example token labels (optional)
    tokens = [f"tok_{i}" for i in range(T)]

    mha = MultiHeadAttentionViz(d_model=d_model, num_heads=H, dropout=0.0)
    out, attn = mha(x, x, x, mask=None, return_attn=True)  # self-attention

    print("Output shape:", out.shape)        # [B, T, d_model]
    print("Attention shape:", attn.shape)    # [B, H, T, S]

    plot_attention_heads(attn, batch_index=0, tokens=tokens, title="Self-Attention Heatmaps per Head")
    plot_attention_for_position(attn, position=3, batch_index=0, tokens=tokens)