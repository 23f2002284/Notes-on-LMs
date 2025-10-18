import torch

def make_causal_mask(T: int, S: int | None = None, batch_size: int = 1, num_heads: int = 1, device=None, dtype=torch.float32):
    """
    Create a causal mask that allows attention only to current and past positions.
    Returns a mask of shape [B, H, T, S] with 1.0 for keep and 0.0 for masked-out.

    Args:
      T: query length
      S: key length (defaults to T if None)
      batch_size: B
      num_heads: H
      device: torch device
      dtype: torch dtype (float recommended for broad compatibility)
    """
    if S is None:
        S = T
    mask_2d = torch.tril(torch.ones((T, S), dtype=dtype, device=device))  # [T, S], 1 below/diag, 0 above
    mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, T, S)  # [B, H, T, S]
    return mask