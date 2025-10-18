"""
Single Token Attention Implementation

This module implements attention mechanism for a single query token against a sequence of key-value pairs.
This is a fundamental building block of transformer models where we compute attention weights for one position at a time.

Key Concepts:
- Query (q): The token we want to compute attention for
- Keys (K): All tokens in the sequence that the query attends to
- Values (V): The actual content being attended to
- Attention scores: How much each key is relevant to the query
- Output: Weighted sum of values based on attention scores
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

def single_token_attention(
    query: torch.Tensor, # shape (batch_size, 1, d_k) where d_k is the dimension of the key
    keys: torch.Tensor, # shape (batch_size, seq_len, d_k)
    values: torch.Tensor, # shape (batch_size, seq_len, d_v) where d_v is the dimension of the value
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Compute attention for a single query position across all sequences in the batch.
    
    Args:
        query: Query tensor of shape [batch_size, 1, d_k]
        keys: Key tensor of shape [batch_size, seq_len, d_k]
        values: Value tensor of shape [batch_size, seq_len, d_v]
        mask: Optional mask tensor of shape [batch_size, 1, seq_len]
        dropout: Optional dropout layer to apply to attention weights
        
    Returns:
        Tuple of (output, attention_weights)
        - output: [batch_size, 1, d_v]
        - attention_weights: [batch_size, 1, seq_len]
    """

    # validation
    assert query.dim() == 3, "Query must be of shape [batch_size, 1, d_k]"
    assert keys.dim() == 3, "Keys must be of shape [batch_size, seq_len, d_k]"
    assert values.dim() == 3, "Values must be of shape [batch_size, seq_len, d_v]"
    assert query.size(0) == keys.size(0) == values.size(0), "Batch sizes must match"
    assert query.size(2) == keys.size(2), "Key and query dimensions must match"
    assert keys.size(1) == values.size(1), "Sequence length must match"

    # compute attention scores
    scores = torch.bmm(query, keys.transpose(-2, -1)) # shape (batch_size, 1, seq_len) = (batch_size, 1, d_k) @ (batch_size, d_k, seq_len)
    
    # scaling by 1/sqrt(d_k)
    scores = scores / math.sqrt(query.size(-1)) # shape (batch_size, 1, seq_len) 

    # apply mask if provided
    if mask is not None:
        mask = mask.to(scores.dtype)
        scores = scores.masked_fill(mask == 0, -1e9)

    # compute attention weights
    weights = F.softmax(scores, dim=-1) # shape (batch_size, 1, seq_len)
    
    # apply dropout if provided
    if dropout is not None:
        weights = dropout(weights)
    
    # compute output = weighted sum of values
    # shape (batch_size, 1, d_v) = (batch_size, 1, seq_len) @ (batch_size, seq_len, d_v)
    output = torch.bmm(weights, values) 
    
    return output, weights

if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_k = 64
    d_v = 64
    
    query = torch.randn(batch_size, 1, d_k)
    keys = torch.randn(batch_size, seq_len, d_k)
    values = torch.randn(batch_size, seq_len, d_v)
    mask = torch.ones(batch_size, 1, seq_len)
    mask[0, 0, -3] = 0 # Mask out last 3 tokens in first sequence
    dropout = nn.Dropout(0.1)
    
    output, weights = single_token_attention(query, keys, values, mask, dropout)
    print(output.shape) # should be [batch_size, 1, d_v] i.e. (32, 1, 64)
    print(weights.shape) # should be [batch_size, 1, seq_len] i.e. (32, 1, 10)