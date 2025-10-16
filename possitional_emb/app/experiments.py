from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from tiny_transformer import TinyCausalTransformer
from tasks import (
    TaskSpec,
    ClassificationHead,
    LMHead,
    batch_modular_sum,
    batch_k_back,
    batch_shift_invariant_pattern,
    batch_long_range_match,
    compute_cls_loss_and_acc,
    compute_lm_loss_and_acc,
)


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 1337
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 1024  # should cover long_seq_len


def set_seed(seed: int):
    import random, os
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model_for_task(
    positional_type: str, spec: TaskSpec, cfg: TrainConfig
) -> Tuple[nn.Module, Optional[nn.Module]]:
    model = TinyCausalTransformer(
        vocab_size=spec.vocab_size,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        positional_type=positional_type,
    )
    head = None
    if spec.kind == "cls":
        # Classes: for modular sum we fix 10; others are binary
        n_classes = 10 if "modular_sum" in spec.name else 2
        head = ClassificationHead(cfg.d_model, n_classes)
    else:
        head = LMHead(cfg.d_model, spec.vocab_size)
    return model, head


def sample_batch(spec: TaskSpec, device: torch.device, train: bool, **kwargs):
    if spec.name.startswith("modular_sum"):
        return batch_modular_sum(kwargs["batch_size"], kwargs["seq_len"], spec.vocab_size, device)
    elif spec.name.startswith("k_back_lm"):
        return batch_k_back(kwargs["batch_size"], kwargs["seq_len"], spec.vocab_size, kwargs["k_back"], device)
    elif spec.name.startswith("shift_invariant_pattern"):
        return batch_shift_invariant_pattern(kwargs["batch_size"], kwargs["seq_len"], spec.vocab_size, device, train=train)
    elif spec.name.startswith("long_range_match"):
        return batch_long_range_match(kwargs["batch_size"], kwargs["seq_len"], spec.vocab_size, device)
    else:
        raise ValueError(f"Unknown task: {spec.name}")


def eval_model(
    model: nn.Module,
    head: nn.Module,
    spec: TaskSpec,
    device: torch.device,
    cfg: TrainConfig,
    seq_len: int,
    k_back: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    head.eval()
    metrics = {"loss": 0.0, "acc": 0.0}
    n_batches = 10
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(
                spec,
                device,
                train=False,
                batch_size=cfg.batch_size,
                seq_len=seq_len,
                k_back=k_back if k_back is not None else 0,
            )
            h, _ = model(x, return_attn=False)
            if spec.kind == "cls":
                logits = head(h)
                loss, acc = compute_cls_loss_and_acc(logits, y)
            else:
                logits = head(h)
                loss, acc = compute_lm_loss_and_acc(logits, y)
            metrics["loss"] += loss.item()
            metrics["acc"] += acc
    for k in metrics:
        metrics[k] /= n_batches
    return metrics


def train_single(
    positional_type: str,
    spec: TaskSpec,
    cfg: TrainConfig,
    k_back: Optional[int] = None,
    capture_attn_example: bool = False,
) -> Dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model, head = build_model_for_task(positional_type, spec, cfg)
    model.to(device)
    head.to(device)

    params = list(model.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    logs = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    model.train()
    head.train()

    attn_example = None
    example_x = None

    for step in range(cfg.steps):
        x, y = sample_batch(
            spec,
            device,
            train=True,
            batch_size=cfg.batch_size,
            seq_len=spec.train_seq_len,
            k_back=k_back if k_back is not None else 0,
        )
        h, _ = model(x, return_attn=False)
        if spec.kind == "cls":
            logits = head(h)
            loss, acc = compute_cls_loss_and_acc(logits, y)
        else:
            logits = head(h)
            loss, acc = compute_lm_loss_and_acc(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        logs["train_loss"].append(loss.item())
        logs["train_acc"].append(acc)

        # quick val
        if (step + 1) % max(1, (cfg.steps // 10)) == 0 or step == cfg.steps - 1:
            val = eval_model(model, head, spec, device, cfg, seq_len=spec.val_seq_len, k_back=k_back)
            logs["val_loss"].append(val["loss"])
            logs["val_acc"].append(val["acc"])

    # final evals
    in_dist = eval_model(model, head, spec, device, cfg, seq_len=spec.val_seq_len, k_back=k_back)
    out_dist = eval_model(model, head, spec, device, cfg, seq_len=spec.long_seq_len, k_back=k_back)

    # capture one attention map on a small example
    if capture_attn_example:
        example_x, _ = sample_batch(
            spec, device, train=False, batch_size=1, seq_len=min(32, spec.val_seq_len), k_back=k_back if k_back else 0
        )
        with torch.no_grad():
            _, attns = model(example_x, return_attn=True)
        # attns is a list of (B,H,T,T) per layer from last block only (we collected all); keep last layer
        if len(attns) > 0:
            attn_example = attns[-1].squeeze(0)  # (H,T,T)
        else:
            attn_example = None

    result = {
        "positional_type": positional_type,
        "train_logs": logs,
        "in_dist": in_dist,
        "out_dist": out_dist,
        "attn_example": attn_example.cpu().numpy() if attn_example is not None else None,
        "example_x": example_x.cpu().numpy() if example_x is not None else None,
    }
    return result


def run_experiments(
    positional_types: List[str],
    spec: TaskSpec,
    cfg: TrainConfig,
    k_back: Optional[int] = None,
    capture_attn_example: bool = False,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in positional_types:
        out[p] = train_single(p, spec, cfg, k_back=k_back, capture_attn_example=capture_attn_example)
    return out