import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils import data
from toy_tasks import get_dataloader

data = get_dataloader(
    task='next_token',
    vocab_size=1000,
    seq_len=10,
    batch_size=32,
    num_samples=100,
    shuffle=True,
    pattern= 'alternating'
)
for src, tgt in data:
    breakpoint()