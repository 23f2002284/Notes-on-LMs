# TODO: for now we will have a simple RoPE but we will also refer to the repo: https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    An implementation of Rotary Positional Embedding (RoPE).

    RoPE encodes position information by rotating pairs of features in the
    input embeddings. This method has become standard in many modern LLMs
    as it naturally encodes relative positions and has good extrapolation properties.

    Args:
        emb_dim (int): The embedding dimension. Must be an even number.
        seq_len (int): The maximum sequence length. Default: 5000.
        n (int): The base for the geometric progression of frequencies. Default: 10000.
    """
    def __init__(self, emb_dim: int, seq_len: int = 5000, n: int = 10000):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, but got {emb_dim}")

        # Precompute the inverse frequencies (the 'theta' term in the paper)
        # Shape: (emb_dim / 2)
        inv_freq = 1.0 / (n ** (torch.arange(0, emb_dim, 2, dtype=torch.float) / emb_dim))
        
        # Precompute the positions
        # Shape: (seq_len)
        t = torch.arange(seq_len, dtype=torch.float)
        
        # Outer product to get frequencies for each position
        # Shape: (seq_len, emb_dim / 2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Concatenate frequencies to handle both sin and cos components
        # This creates the 'm * theta' matrix from the paper
        # Shape: (seq_len, emb_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Register sin and cos values as buffers. They are not trainable parameters.
        # We reshape them to (1, seq_len, 1, emb_dim) to make broadcasting
        # with the input tensor (batch, seq_len, heads, head_dim) easy.
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :])
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :])

    def _apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotary rotation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_heads, head_dim).

        Returns:
            torch.Tensor: Rotated tensor.
        """
        # Reshape input to group features into pairs
        # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads, head_dim/2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        
        # Get the even and odd indexed features
        x1 = x_pairs[..., 0]  # Real part
        x2 = x_pairs[..., 1]  # Imaginary part

        # This is the clever trick to apply the rotation matrix:
        # [cos, -sin]
        # [sin,  cos]
        # The 'rotated' vector is [-x2, x1], which corresponds to a 90-degree rotation.
        rotated_x = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
        
        # Element-wise multiplication with the precomputed sin/cos values
        # This is equivalent to the complex number multiplication: (x1 + ix2) * (cos + isin)
        seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len, ...]
        sin = self.sin_cached[:, :seq_len, ...]
        
        rotated_tensor = x * cos + rotated_x * sin
        return rotated_tensor.type_as(x)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to Query and Key tensors.

        Args:
            query (torch.Tensor): Query tensor from self-attention.
            key (torch.Tensor): Key tensor from self-attention.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The rotated Query and Key tensors.
        """
        return self._apply_rotary_emb(query), self._apply_rotary_emb(key)

