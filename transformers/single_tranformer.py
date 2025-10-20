"""
Transformer Block Implementation

Combines:
- Multi-head self-attention
- Layer normalization
- Residual connections
- Position-wise feed-forward network

Architecture (Pre-Norm):
  x → LayerNorm → MultiHeadAttention → residual → 
  → LayerNorm → FeedForward → residual → output

Architecture (Post-Norm):
  x → MultiHeadAttention → residual → LayerNorm → 
  → FeedForward → residual → LayerNorm → output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    Two linear transformations with a nonlinearity in between

    FFN(x) = max(0, xW1 + b1)W2 + b2 ( if using ReLU )
    or
    FFN(x) = (xW1 + b1)W2 + b2 ( if using GELU )
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    A single transformer encoder block with:
    - Pre-LayerNorm architecture
    - Multi-head self attention
    - Residual connection
    - Position-wise feed forward network

    Forward flow ( pre-norm ):
        x_norm = LayerNorm(x)
        attn_out = MultiHeadAttention(x_norm)
        x = x + attn_out  # residual connection
        x_norm = LayerNorm(x)
        ffn_out = FeedForward(x_norm)
        x = x + ffn_out # residual connection
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        from attention import MultiHeadAttentionViz as MultiHeadAttention

        self.attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout
        )

        self.FeedForward = FeedForward(
            d_model = d_model,
            d_ff = d_ff,
            dropout = dropout,
            activation = activation
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
            return_attn: bool, if true, return (output, attention_weights)
        Returns:
            [batch_size, seq_len, d_model]
        """

        # Self-attention block with pre-norm and residual

        x_norm =self.norm1(x)

        if return_attn:
            attn_out, attn_weights = self.attention(
                x_norm, x_norm, x_norm,
                mask = mask,
                return_attn = True
            )
        else:
            attn_out = self.attention(
                x_norm, x_norm, x_norm,
                mask = mask
            )
            attn_weights = None
        
        x = x + self.dropout(attn_out) # residual connection

        # Feed Forward block with pre-norm and residual
        x_norm = self.norm2(x)
        ff_out = self.FeedForward(x_norm)
        x = x + ff_out   # as already dropout inside feed forward class
        
        if return_attn:
            return x, attn_weights
        return x
        

class TransformerBlockPostNorm(nn.Module):
    """
    Post-LayerNorm variant (original "Attention is All You Need").
    Generally less stable than pre-norm, but included for completeness.
    
    Forward flow (post-norm):
        attn_out = MultiHeadAttention(x, x, x, mask)
        x = LayerNorm(x + attn_out)
        
        ff_out = FeedForward(x)
        x = LayerNorm(x + ff_out)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        from notes.attention import MultiHeadAttnViz as MultiHeadAttention

        self.attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout
        )

        self.FeedForward = FeedForward(
            d_model = d_model,
            d_ff = d_ff,
            dropout = dropout,
            activation = activation
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
            return_attn: bool, if true, return (output, attention_weights)
        Returns:
            [batch_size, seq_len, d_model]
        """

        # Self-attention block with post-norm

        if return_attn:
            attn_out, attn_weights = self.attention(
                x, x, x,
                mask = mask,
                return_attn = True
            )
        else:
            attn_out = self.attention(
                x, x, x,
                mask = mask
            )
            attn_weights = None

        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward block with post-norm
        ff_out = self.FeedForward(x)
        x = self.norm2(x + ff_out)

        if return_attn:
            return x, attn_weights
        return x