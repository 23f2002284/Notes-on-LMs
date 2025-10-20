import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import Optional

from mini_tranformers import MiniTransformer
from toy_tasks import get_dataloader
from attention import make_causal_mask

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_casual_mask: bool = False
):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for src, tgt in tqdm(dataloader, desc = "Training", leave= False):
        src, tgt = src.to(device), tgt.to(device)
        B, T = src.shape

        optimizer.zero_grad()

        # Build Casual mask if needed
        mask = None
        if use_casual_mask:
            num_heads = model.blocks[0].attention.num_heads
            mask = make_causal_mask(T, T, B, num_heads, device)

        # forward
        logits = model(src, mask = mask, return_attn = False) # [B, T, vocab_size]

        # Compute loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == tgt).sum().item()
        total += tgt.numel()
    
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_casual_mask: bool = False
):
    """
    Evaluate the model
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        B, T = src.shape

        mask = None
        if use_casual_mask:
            num_heads = model.blocks[0].attention.num_heads
            mask = make_causal_mask(T, T, B, num_heads, device)
        
        # forward
        logits = model(src, mask = mask, return_attn = False) # [B, T, vocab_size]
        
        # Compute loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        
        # Stats
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == tgt).sum().item()
        total += tgt.numel()
    
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc

def plot_training_curves(
    train_losses: list[float],
    train_accs: list[float],
    val_losses: list[float],
    val_accs: list[float],
    save_path: Optional[str] = None
):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig