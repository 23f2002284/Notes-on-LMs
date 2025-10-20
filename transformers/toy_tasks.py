"""
Toy tasks for testing mini-transformer:
1. Copy task: copy input sequence
2. Reverse task: reverse input sequence  
3. Next token prediction: simple language modeling
"""

import torch
from torch.utils.data import DataLoader, Dataset
import random
from typing import Optional

class CopyTask(Dataset):
    """
    Copy Task: Given sequence [ a, b, c ], predict [a, b, c]
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(
        self
    ):
        return self.num_samples
    
    def __getitem__(self, idx):
        # generate random sequence ( exclusing 0, while we reserve for padding )
        seq = torch.randint(1, self.vocab_size, (self.seq_len,))
        # target is same as input 
        return seq, seq.clone()

    
class ReverseTask(Dataset):
    """
    Reverse Task: Given [a, b, c], predict [c, b, a]
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.randint(1, self.vocab_size, (self.seq_len,))
        target = torch.flip(seq, [0])
        return seq, target
    

class NextTokenTaskrandom(Dataset):
    """
    Next token prediction: Given [a, b, c], predict [ b, c, d]
    Simple auto-regressive task
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples

    def __getitem__(
        self, idx
    ):
        # generate sequence of length seq_len + 1
        seq = torch.randint(1, self.vocab_size, (self.seq_len+1,))
        return seq[:-1], seq[1:]

class NextTokenTaskLearnable(Dataset):
    """
    Next token prediction with LEARNABLE patterns.
    Creates sequences like: [1, 2, 3, 4] → [2, 3, 4, 5] (increment by 1)
    Or modulo patterns: [1, 2, 3, 4] → [2, 3, 4, 0] (mod vocab_size)
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, pattern: str = 'increment'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pattern = pattern
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.pattern == 'increment':
            # Arithmetic sequence: [a, a+1, a+2, ...] → [a+1, a+2, a+3, ...]
            start = torch.randint(0, self.vocab_size, (1,)).item()
            seq = torch.tensor([(start + i) % self.vocab_size for i in range(self.seq_len + 1)])
        
        elif self.pattern == 'repeat':
            # Repeating pattern: [1, 2, 3, 1, 2, 3, ...] → [2, 3, 1, 2, 3, 1, ...]
            pattern_len = min(5, self.vocab_size)
            base_pattern = torch.randint(1, self.vocab_size, (pattern_len,))
            seq = base_pattern.repeat((self.seq_len + 1) // pattern_len + 1)[:self.seq_len + 1]
        
        elif self.pattern == 'reverse_copy':
            # Copy then reverse: [1, 2, 3] → [3, 2, 1]
            first_half = torch.randint(1, self.vocab_size, (self.seq_len // 2 + 1,))
            seq = torch.cat([first_half, torch.flip(first_half, [0])])[:self.seq_len + 1]
        
        else:  # 'modulo' - simple counting with wraparound
            start = torch.randint(0, self.vocab_size, (1,)).item()
            seq = torch.tensor([(start + i) % self.vocab_size for i in range(self.seq_len + 1)])
        
        return seq[:-1], seq[1:]  # Input: first seq_len, Target: last seq_len


class NextTokenTaskLearnableComplex(Dataset):
    """
    Next token prediction with learnable patterns that challenge transformer models.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, pattern: str = 'fibonacci'):
        self.vocab_size = max(vocab_size, 10)  # Ensure minimum vocab size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pattern = pattern
        self.primes = self._sieve_of_eratosthenes(self.vocab_size * 2)  # Pre-compute primes
    
    def _sieve_of_eratosthenes(self, n):
        """Generate primes up to n using Sieve of Eratosthenes"""
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i :: i] = [False] * len(sieve[i*i :: i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def __len__(self):
        return self.num_samples

    def _generate_fibonacci(self, length):
        """Generate Fibonacci-like sequence with modulo to stay in vocab"""
        a, b = torch.randint(1, min(10, self.vocab_size), (2,))
        seq = [a, b]
        for _ in range(length - 2):
            a, b = b, (a + b) % self.vocab_size
            seq.append(b)
        return torch.tensor(seq)

    def _generate_primes(self, length):
        """Generate sequence of primes with some variations"""
        prime_seq = []
        current = torch.randint(1, 10, (1,)).item()
        for _ in range(length):
            # Find next prime greater than current
            next_p = next((p for p in self.primes if p > current), self.primes[0])
            prime_seq.append(next_p % self.vocab_size)
            current = next_p
        return torch.tensor(prime_seq)

    def _generate_alternating(self, length):
        """Generate sequence with alternating operations"""
        x = torch.randint(1, self.vocab_size, (1,)).item()
        seq = [x]
        for i in range(1, length):
            if i % 3 == 0:
                x = (x + 3) % self.vocab_size
            elif i % 2 == 0:
                x = (x * 2) % self.vocab_size
            else:
                x = max(1, (x - 1) % self.vocab_size)
            seq.append(x)
        return torch.tensor(seq)

    def _generate_multiplicative(self, length):
        """Generate sequence with multiplicative patterns"""
        x = torch.randint(1, min(5, self.vocab_size // 2), (1,)).item()
        seq = [x]
        for _ in range(length - 1):
            x = (x * 2) % self.vocab_size
            if x == 0:  # Avoid 0 in the sequence
                x = 1
            seq.append(x)
        return torch.tensor(seq)

    def _generate_random_walk(self, length):
        """Generate sequence with random walk pattern"""
        x = torch.randint(1, self.vocab_size, (1,)).item()
        seq = [x]
        for _ in range(length - 1):
            step = torch.randint(-2, 3, (1,)).item()
            x = max(1, min(self.vocab_size - 1, x + step))
            seq.append(x)
        return torch.tensor(seq)

    def _generate_skip_pattern(self, length):
        """Generate sequence with skip patterns"""
        pattern = torch.randint(1, min(5, self.vocab_size), (2,))
        seq = []
        for i in range(length):
            if i % 3 == 0:
                seq.append(pattern[0])
            else:
                seq.append((pattern[0] + pattern[1] * (i % 2)) % self.vocab_size)
        return torch.tensor(seq)

    def __getitem__(self, idx):
        if self.pattern == 'fibonacci':
            seq = self._generate_fibonacci(self.seq_len + 1)
        elif self.pattern == 'primes':
            seq = self._generate_primes(self.seq_len + 1)
        elif self.pattern == 'alternating':
            seq = self._generate_alternating(self.seq_len + 1)
        elif self.pattern == 'multiplicative':
            seq = self._generate_multiplicative(self.seq_len + 1)
        elif self.pattern == 'random_walk':
            seq = self._generate_random_walk(self.seq_len + 1)
        elif self.pattern == 'skip':
            seq = self._generate_skip_pattern(self.seq_len + 1)
        else:
            # Default to simple increment if pattern not recognized
            start = torch.randint(0, self.vocab_size, (1,)).item()
            seq = torch.tensor([(start + i) % self.vocab_size for i in range(self.seq_len + 1)])

        return seq[:-1].long(), seq[1:].long()  # Ensure tensors are long type
    

def get_dataloader(
    task:str,
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    batch_size: int,
    shuffle: bool = True,
    pattern: Optional[str] = None
):
    """
    Get dataloader for a toy task.
    
    Args:
        task: 'copy', 'reverse', or 'next_token'
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        num_samples: Number of samples
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    if task == 'copy':
        dataset = CopyTask(
            vocab_size,
            seq_len,
            num_samples
        )
    elif task == 'reverse':
        dataset = ReverseTask(
            vocab_size,
            seq_len,
            num_samples
        )
    elif task == 'next_token':
        dataset = NextTokenTaskLearnableComplex(
            vocab_size,
            seq_len,
            num_samples,
            pattern=pattern
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return DataLoader(dataset, batch_size=batch_size, shuffle = shuffle)