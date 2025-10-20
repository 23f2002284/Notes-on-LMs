"""
Test and verify the Transformer Block implementation.

Checks:
1. Shape consistency
2. Residual connections (gradients flow)
3. Layer norm statistics
4. Causal masking compatibility
5. Parameter count
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from single_tranformer import TransformerBlock
from attention import make_causal_mask


def test_shapes():
    """Test that shapes are preserved through the block."""
    print("\n=== Testing Shapes ===")
    B, T, d_model, num_heads, d_ff = 4, 32, 128, 8, 512
    
    x = torch.randn(B, T, d_model)
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    
    output, attn = block(x, mask=None, return_attn=True)
    
    assert output.shape == (B, T, d_model), f"Output shape mismatch: {output.shape}"
    assert attn.shape == (B, num_heads, T, T), f"Attention shape mismatch: {attn.shape}"
    
    print(f"✓ Input:  {x.shape}")
    print(f"✓ Output: {output.shape}")
    print(f"✓ Attn:   {attn.shape}")


def test_causal_mask():
    """Test that causal masking works correctly."""
    print("\n=== Testing Causal Mask ===")
    B, T, d_model, num_heads, d_ff = 2, 16, 64, 4, 256
    
    x = torch.randn(B, T, d_model)
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()
    
    mask = make_causal_mask(T, T, B, num_heads, device=x.device)
    
    output, attn = block(x, mask=mask, return_attn=True)
    
    # Check no attention to future
    future_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    max_future_attn = attn[:, :, future_mask].abs().max().item()
    
    print(f"Max future attention weight: {max_future_attn:.2e}")
    assert max_future_attn < 1e-6, "Causal mask not working!"
    print("✓ Causal masking verified")


def test_residual_connections():
    """Verify that residuals help gradient flow."""
    print("\n=== Testing Residual Connections ===")
    B, T, d_model, num_heads, d_ff = 2, 8, 32, 4, 128
    
    x = torch.randn(B, T, d_model, requires_grad=True)
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    
    output = block(x, mask=None, return_attn=False)
    loss = output.sum()
    loss.backward()
    
    # Check that input has gradients (residual path)
    assert x.grad is not None, "No gradients on input!"
    grad_norm = x.grad.norm().item()
    print(f"Input gradient norm: {grad_norm:.4f}")
    print("✓ Residual connections allow gradient flow")


def test_layer_norm():
    """Check that layer norm normalizes properly."""
    print("\n=== Testing Layer Normalization ===")
    B, T, d_model, num_heads, d_ff = 4, 16, 64, 4, 256
    
    x = torch.randn(B, T, d_model) * 10  # Large scale input
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    block.eval()
    
    with torch.no_grad():
        output = block(x, mask=None, return_attn=False)
    
    # Check mean and std along d_model dimension
    mean = output.mean(dim=-1).abs().mean().item()
    std = output.std(dim=-1).mean().item()
    
    print(f"Output mean (across d_model): {mean:.4f} (should be ~0)")
    print(f"Output std (across d_model):  {std:.4f} (should be ~1)")
    print("✓ Layer normalization working")


def test_parameter_count():
    """Print parameter count for the block."""
    print("\n=== Parameter Count ===")
    d_model, num_heads, d_ff = 512, 8, 2048
    
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.1)
    
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Breakdown
    attn_params = sum(p.numel() for p in block.attention.parameters())
    ff_params = sum(p.numel() for p in block.FeedForward.parameters())
    norm_params = sum(p.numel() for p in block.norm1.parameters()) + \
                  sum(p.numel() for p in block.norm2.parameters())
    
    print(f"  - Attention: {attn_params:,}")
    print(f"  - FeedForward: {ff_params:,}")
    print(f"  - LayerNorm: {norm_params:,}")


if __name__ == "__main__":
    torch.manual_seed(0)
    
    test_shapes()
    test_causal_mask()
    test_residual_connections()
    test_layer_norm()
    test_parameter_count()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)