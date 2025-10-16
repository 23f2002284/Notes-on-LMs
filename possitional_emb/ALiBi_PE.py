import torch
import torch.nn as nn
import math

class ALiBiPositionalEncoding(nn.Module):
    """
    An implementation of Attention with Linear Biases (ALiBi).

    ALiBi is a method for encoding position information that adds a static,
    non-learned bias to the attention scores. This bias is proportional to the
    distance between the query and key tokens, providing a locality bias.

    Args:
        num_heads (int): The number of attention heads.
        seq_len (int): The maximum sequence length to precompute the bias for.
    """
    def __init__(self, num_heads: int, seq_len: int):
        super().__init__()
        self.num_heads = num_heads

        # Precompute the slopes 'm' for each head.
        # The slopes are in a geometric progression, as suggested in the paper.
        # e.g., for 8 heads: 2**(-8/8), 2**(-7/8), ..., 2**(-1/8)
        slopes = torch.Tensor(self._get_slopes(num_heads))

        # Precompute the distance matrix |i - j|.
        # The arange calls create the row and column indices.
        # The absolute difference gives the distance.
        positions = torch.arange(seq_len).unsqueeze(1)
        distances = torch.abs(positions - positions.T)

        # Multiply slopes by distances to get the final bias matrix.
        # The shape is (num_heads, seq_len, seq_len).
        alibi_bias = distances.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(2)

        # Register 'alibi_bias' as a buffer. It's part of the model's state
        # but not a trainable parameter.
        self.register_buffer("alibi_bias", -alibi_bias)

    @staticmethod
    def _get_slopes(n: int) -> list[float]:
        """Calculates the geometric progression of slopes for each head."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            # This is a simpler version if num_heads is not a power of 2.
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2) +
                get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
            )

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Adds the ALiBi bias to the attention scores.

        Args:
            attn_scores (torch.Tensor): The raw attention scores from (Q @ K.T),
                                        with shape (batch_size, num_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: The attention scores with the ALiBi bias added.
        """
        # Get the sequence length from the input scores.
        seq_len = attn_scores.size(-1)

        # Slice the precomputed bias to match the input sequence length and add it.
        # Broadcasting handles the addition across the batch dimension.
        # Bias shape: (1, num_heads, seq_len, seq_len)
        # Scores shape: (batch_size, num_heads, seq_len, seq_len)
        bias_slice = self.alibi_bias[:, :seq_len, :seq_len].unsqueeze(0)
        return attn_scores + bias_slice