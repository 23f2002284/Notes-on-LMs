import torch
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Optional, Literal
from single_tranformer import TransformerBlock
import os
import sys
import math 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possitional_emb import SinusoidalPositionalEncoding, LearnedPositionalEmbedding, RoPE, ALiBiPositionalEncoding

class MiniTransformer(nn.Module):
    """
    A small transformer model for sequence-to-sequence tasks.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        num_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        pos_encoding: Type of positional encoding ('sinusoidal' or 'learned')
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pos_encoding: Literal['sinusoidal', 'learned'] = 'sinusoidal',
        tie_weights: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Token Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # Position encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout, max_seq_len)
        elif pos_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEmbedding(d_model, max_seq_len)
        else:
            raise ValueError(f"Unknown positional encoding: {pos_encoding}")
        
        # Stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                dropout = dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Optional weight tying (share embedding and output weight)
        if tie_weights:
            self.output_projection.weight = self.token_embeddings.weight

        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ):
        """
        Args:
            x: [batch_size, seq_len]
            mask: [batch_size, seq_len, seq_len] or None
            return_attn: bool, if true, return (output, attention_weights)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            attn_weights (Optional): List of [batch_size, num_heads, seq_len, seq_len]
        """

        # token embedding and positional embeddings
        x = self.token_embeddings(x) * math.sqrt(self.d_model) # Scale by sqrt (d_model)
        x = self.pos_encoding(x) # [batch_size, seq_len, d_model]

        # store attention weights if requested
        attn_weights_list = []

        # pass through transformer blocks
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, mask=mask, return_attn = True)
                attn_weights_list.append(attn)
            else:
                x = block(x, mask=mask)
        
        # output projection
        logits = self.output_projection(x)

        if return_attn:
            return logits, attn_weights_list
        return logits

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        