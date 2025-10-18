import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttentionViz(nn.Module):
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
    
    # Replace your _prepare_mask with this more robust version
    @staticmethod
    def _prepare_mask(mask: torch.Tensor, B: int, H: int, T: int, S: int, device, dtype) -> torch.Tensor:
        """
        Normalize various mask shapes to [B, H, T, S] boolean mask where True=keep.
        Accepts:
        - [B, S]
        - [B, 1, S]
        - [B, T, S]
        - [B, 1, T, S]
        - [B, H, T, S]
        If a 4D mask is larger than needed, it is sliced to [:T, :S].
        If smaller and not broadcastable, raises a helpful error.
        """
        if mask is None:
            return None

        # Convert to boolean keep-mask
        if mask.dtype != torch.bool:
            mask = mask != 0
        mask = mask.to(device=device)

        if mask.dim() == 2:
            # [B, S] -> [B,1,1,S]
            if mask.shape != (B, S):
                raise ValueError(f"2D mask must be [B,S]={B,S}, got {tuple(mask.shape)}")
            mask = mask[:, None, None, :]  # [B,1,1,S]

        elif mask.dim() == 3:
            # [B,1,S] or [B,T,S]
            if mask.shape[1] == 1:
                if mask.shape != (B, 1, S):
                    raise ValueError(f"3D mask [B,1,S] expected {B,1,S}, got {tuple(mask.shape)}")
                mask = mask[:, :, None, :]  # [B,1,1,S]
            else:
                if mask.shape != (B, T, S):
                    raise ValueError(f"3D mask [B,T,S] expected {B,T,S}, got {tuple(mask.shape)}")
                mask = mask[:, None, :, :]  # [B,1,T,S]

        elif mask.dim() == 4:
            b, h, t, s = mask.shape
            if b not in (1, B):
                raise ValueError(f"4D mask batch dim must be 1 or {B}, got {b}")
            if h not in (1, H):
                raise ValueError(f"4D mask head dim must be 1 or {H}, got {h}")

            # If larger than needed, slice down; if smaller, try broadcast if size==1
            if t != T or s != S:
                if t >= T and s >= S:
                    mask = mask[..., :T, :S]
                else:
                    # If either t or s is 1, it can broadcast; otherwise raise
                    if t not in (1, T) or s not in (1, S):
                        raise ValueError(
                            f"4D mask last dims must be broadcastable or >= target. "
                            f"Got {t,s}, need {T,S}. Consider rebuilding the mask for current seq_len."
                        )
        else:
            raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}. Expected 2D/3D/4D.")

        # Finally broadcast to [B, H, T, S]
        mask = mask.expand(B, H, T, S)
        return mask


        
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the multi-head attention module for vizualization
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: Optional, broadcastable to [batch_size, num_heads, seq_len, key_seq_len] or boolean ( True = keep )
            return_attn: bool, if True, also return attention weights [batch_size, num_heads, seq_len, key_seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attn_weights: (Optional)[batch_size, num_heads, seq_len, key_seq_len]
        """
        batch_size, seq_len, _ = query.size()
        key_seq_len = key.size(1)
        
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
        # [batch_size, num_heads, seq_len, d_k] @ [batch_size, num_heads, d_k, key_seq_len] -> [batch_size, num_heads, seq_len, key_seq_len]
        scale = 1 / math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # apply mask if provided
        if mask is not None:
            mask_b = MultiHeadAttentionViz._prepare_mask(mask, batch_size, self.num_heads, seq_len, key_seq_len, scores.device, scores.dtype)
            scores = scores.masked_fill(~mask_b, float('-inf'))

        # compute attention weights 
        attn_weights = F.softmax(scores, dim=-1) # shape (batch_size, num_heads, seq_len, key_seq_len)
        
        # for visualization
        self.attn_weights = attn_weights.detach()

        attn_weights = self.dropout(attn_weights) # shape (batch_size, num_heads, seq_len, key_seq_len)
        
        # compute output
        output = torch.matmul(attn_weights, v) # shape (batch_size, num_heads, seq_len, d_v)

        output = output.transpose(1, 2) # shape (batch_size, seq_len, num_heads, d_v)
        output = output.contiguous().view(batch_size, -1, self.d_model) # shape (batch_size, seq_len, d_model)
        # contiguous() is used to ensure that the memory layout of the tensor is contiguous
        # view() is used to change the shape of the tensor

        output = self.W_o(output) # shape (batch_size, seq_len, d_model)
        if return_attn:
            return output, attn_weights
        return output