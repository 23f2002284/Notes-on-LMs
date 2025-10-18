import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """
    def __init__(
        self,
        d_model: int, # embedding dimension
        num_heads: int, # number of attention heads
        dropout: float = 0.1,
    ):
        super().__init__()
        # validation 
        # as d_model is the dimension of the input and output of the multi-head attention, it must be divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # linear layers for q, k, v
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # output projection
        self.W_o = nn.Linear(d_model, d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Input shape: [batch_size, seq_len, d_model]
        Output shape: [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # shape (batch_size, num_heads, seq_len, d_k) and num_heads * d_k = d_model
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the multi-head attention module
        """
        batch_size = query.size(0)
        
        # linear projections and split into heads
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        # as W_q shape is (d_model, d_model) and query shape is (batch_size, seq_len, d_model)
        # (batch_size, seq_len, d_model) @ (d_model, d_model) -> (batch_size, seq_len, d_model)
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        
        # split into heads
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # compute attention scores
        # [batch_size, num_heads, seq_len, d_k] @ [batch_size, num_heads, d_k, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # apply mask if provided
        if mask is not None:
            mask = mask.to(scores.dtype)
            scores = scores.masked_fill(mask == 0, -1e9)

        # compute attention weights 
        attn_weights = F.softmax(scores, dim=-1) # shape (batch_size, num_heads, seq_len, seq_len)
        
        # for visualization
        self.attn_weights = attn_weights

        attn_weights = self.dropout(attn_weights) # shape (batch_size, num_heads, seq_len, seq_len)
        
        # compute output
        output = torch.matmul(attn_weights, v) # shape (batch_size, num_heads, seq_len, d_v)

        output = output.transpose(1, 2) # shape (batch_size, seq_len, num_heads, d_v)
        output = output.contiguous().view(batch_size, -1, self.d_model) # shape (batch_size, seq_len, d_model)
        # contiguous() is used to ensure that the memory layout of the tensor is contiguous
        # view() is used to change the shape of the tensor

        output = self.W_o(output) # shape (batch_size, seq_len, d_model)

        return output