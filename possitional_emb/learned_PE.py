import torch
import torch.nn as nn


# TODO: Adding scaling factor for the positional embedding
# TODO: layer normalization after adding positional embeddings
# TODO: add option to learn relative positions


class LearnedPositionalEmbedding(nn.Module):
    """
    A module for learned, rather than sinusoidal, positional embeddings.

    This module creates a standard PyTorch Embedding layer that is trained along with
    the rest of the model. It learns the optimal representation for each position.

    Args:
        emb_dim (int): The embedding dimension.
        dropout (float): The dropout rate. Default: 0.1.
        seq_len (int): The maximum sequence length. Default: 5000.
    """
    def __init__(self, emb_dim: int, dropout: float = 0.1, seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # This is the core of the learned positional embedding.
        # It's a simple lookup table where the key is the position index
        # and the value is a learned vector of size emb_dim.
        self.position_embedding = nn.Embedding(seq_len, emb_dim)

        # We register 'positions' as a buffer so it's part of the model's state,
        # but not a trainable parameter. This avoids recreating it on every forward pass.
        self.register_buffer('positions', torch.arange(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Injects positional information into the input tensor.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The output tensor with positional information added.
        """
        # Get the actual sequence length from the input tensor.
        seq_len = x.size(1)

        # Get the positional embeddings for the positions [0, 1, ..., seq_len-1].
        # The lookup returns a tensor of shape (seq_len, emb_dim).
        pos_embeddings = self.position_embedding(self.positions[:seq_len])
        
        # Add the positional embeddings to the input tensor.
        # PyTorch's broadcasting automatically handles the addition across the batch dimension.
        # x shape: (batch_size, seq_len, emb_dim)
        # pos_embeddings shape: (seq_len, emb_dim) -> broadcasts to (1, seq_len, emb_dim)
        x = x + pos_embeddings
        
        return self.dropout(x)

