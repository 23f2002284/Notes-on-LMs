from dataclasses import dataclass
from typing import Literal, Tuple, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn


TaskType = Literal["lm", "cls"]


@dataclass
class TaskSpec:
    name: str
    kind: TaskType
    vocab_size: int
    description: str
    train_seq_len: int
    val_seq_len: int
    long_seq_len: int


def make_modular_sum_task(
    vocab_size: int = 20,  # use small digits range
    train_seq_len: int = 64,
    val_seq_len: int = 64,
    long_seq_len: int = 256,
) -> TaskSpec:
    """
    Classification: predict (sum(tokens) % 10).
    Highlights: extrapolation to longer seq_len; learned absolute often fails beyond trained positions.
    """
    return TaskSpec(
        name="modular_sum_cls",
        kind="cls",
        vocab_size=vocab_size,
        description="Predict sum(x) % 10 for the sequence.",
        train_seq_len=train_seq_len,
        val_seq_len=val_seq_len,
        long_seq_len=long_seq_len,
    )


def make_k_back_lm_task(
    vocab_size: int = 50,
    k_back: int = 3,
    train_seq_len: int = 64,
    val_seq_len: int = 64,
    long_seq_len: int = 128,
) -> Tuple[TaskSpec, int]:
    """
    LM: next token equals token from k steps back (for t>=k).
    Highlights: locality bias; ALiBi tends to help.
    """
    spec = TaskSpec(
        name=f"k_back_lm_k{k_back}",
        kind="lm",
        vocab_size=vocab_size,
        description=f"Predict x[t] = x[t-{k_back}] for t>=k.",
        train_seq_len=train_seq_len,
        val_seq_len=val_seq_len,
        long_seq_len=long_seq_len,
    )
    return spec, k_back


def make_shift_invariant_pattern_task(
    vocab_size: int = 50,
    train_seq_len: int = 64,
    val_seq_len: int = 64,
    long_seq_len: int = 256,
) -> TaskSpec:
    """
    Classification: label=1 if special pair (A,B) occurs anywhere. Train near-start positions, test near-end.
    Highlights: relative position generalization; RoPE/ALiBi tend to help.
    """
    return TaskSpec(
        name="shift_invariant_pattern_cls",
        kind="cls",
        vocab_size=vocab_size,
        description="Detect AB pattern anywhere (train near start, test near end).",
        train_seq_len=train_seq_len,
        val_seq_len=val_seq_len,
        long_seq_len=long_seq_len,
    )


def make_long_range_match_task(
    vocab_size: int = 50,
    train_seq_len: int = 64,
    val_seq_len: int = 64,
    long_seq_len: int = 256,
) -> TaskSpec:
    """
    Classification: label=1 if first token equals last token (balanced).
    Highlights: long-range dependency; locality bias can hurt (ALiBi).
    """
    return TaskSpec(
        name="long_range_match_cls",
        kind="cls",
        vocab_size=vocab_size,
        description="Classify if first token equals last token.",
        train_seq_len=train_seq_len,
        val_seq_len=val_seq_len,
        long_seq_len=long_seq_len,
    )


# -------- Data generation and losses --------

def batch_modular_sum(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    y = (x.sum(dim=1) % 10).long()  # 10 classes
    return x, y


def batch_k_back(
    batch_size: int, seq_len: int, vocab_size: int, k_back: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    # Targets: y[t] = x[t-k] for t>=k else ignore (-100)
    # t-> seq_len and k-> k_back
    y = torch.full((batch_size, seq_len), fill_value=-100, device=device, dtype=torch.long)
    if k_back < seq_len:
        y[:, k_back:] = x[:, :-k_back]
    return x, y


def batch_shift_invariant_pattern(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device, train: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reserve last two IDs for A, B
    A = vocab_size - 2
    B = vocab_size - 1

    x = torch.randint(low=0, high=vocab_size - 2, size=(batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Insert AB with p=0.5 at constrained positions
    for i in range(batch_size):
        if torch.rand((), device=device) < 0.5:
            if train:
                pos = torch.randint(2, min(10, seq_len - 2), (1,), device=device).item()
            else:
                pos = torch.randint(max(2, seq_len - 10), seq_len - 2, (1,), device=device).item()
            x[i, pos] = A
            x[i, pos + 1] = B
            y[i] = 1
    return x, y


def batch_long_range_match(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        if torch.rand((), device=device) < 0.5:
            x[i, -1] = x[i, 0]
            y[i] = 1
        else:
            # Ensure last differs
            diff = torch.randint(low=0, high=vocab_size - 1, size=(1,), device=device)
            x[i, -1] = (x[i, 0] + diff) % vocab_size
            if x[i, -1] == x[i, 0]:
                x[i, -1] = (x[i, -1] + 1) % vocab_size
    return x, y


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, h: torch.Tensor):
        # Use last token representation
        return self.fc(h[:, -1, :])


def compute_cls_loss_and_acc(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(dim=-1) == y).float().mean().item()
    return loss, acc


def compute_lm_loss_and_acc(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
    # logits: (B,T,V), y: (B,T), ignore index -100
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    # Accuracy only on valid positions
    valid = y != -100
    if valid.any():
        preds = logits.argmax(dim=-1)
        acc = (preds[valid] == y[valid]).float().mean().item()
    else:
        acc = 0.0
    return loss, acc


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, h: torch.Tensor):
        return self.fc(h)