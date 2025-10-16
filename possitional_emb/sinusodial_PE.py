import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """A drop-in, efficient sinusoidal positional encoding module.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies, as described in "Attention Is All You Need".
    Applies dropout to the combined embeddings and positional encodings.

    Args:
        emb_dim (int): The embedding dimension.
        dropout (float): The dropout rate. Default: 0.1.
        seq_len (int): The maximum sequence length to precompute for. Default: 5000.
    """
    def __init__(self, emb_dim: int, dropout: float = 0.1, seq_len: int = 5000):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even, but got {emb_dim}")

        self.dropout = nn.Dropout(p=dropout)

        # Create the position and div_term vectors for the formula
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))

        # Precompute the positional encoding matrix (pe)
        pe = torch.zeros(seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer. It's part of the model's state but not a parameter.
        # The shape will be (1, seq_len, emb_dim) to allow for easy broadcasting with the batch dim.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Injects positional information into the input tensor.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, seq_len, emb_dim).

        Returns:
            torch.Tensor: The output tensor with positional information added.
        """
        # x.size(1) is the sequence length of the input batch
        # Add the positional encoding to the input tensor.
        # Slicing self.pe is necessary for inputs shorter than max seq_len.
        # self.pe is (1, max_seq_len, emb_dim)
        # x is (batch_size, seq_len, emb_dim)
        # Broadcasting handles the addition across the batch dimension.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
