"""Training + scoring in bits/letter (base-2 cross-entropy)."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .data import encode
from .model import LetterGPT

LN2 = math.log(2.0)


class CharSeq(Dataset):
    def __init__(self, text: str, context: int):
        self.ids = torch.tensor(encode(text), dtype=torch.long)
        self.context = context

    def __len__(self):
        return max(0, len(self.ids) - self.context - 1)

    def __getitem__(self, i):
        x = self.ids[i:i + self.context]
        y = self.ids[i + 1:i + self.context + 1]
        return x, y


def bits_per_letter(model: LetterGPT, text: str, context: int = 64, batch: int = 256,
                    device: str = "cpu", max_batches: int = 40) -> float:
    """Average surprise: mean -log2 p(true char). Lower is better."""
    model.eval()
    ds = CharSeq(text, context)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)
    tot_nats, tot_tok = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(dl):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            nats = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                   reduction="sum").item()
            tot_nats += nats
            tot_tok += y.numel()
    return (tot_nats / max(tot_tok, 1)) / LN2


def uniform_baseline() -> float:
    import math
    return math.log2(27)


def unigram_baseline(text: str) -> float:
    import math
    from collections import Counter
    ids = encode(text)
    c = Counter(ids)
    n = len(ids)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def cosine_lr(step: int, total: int, peak: float, warmup: int = 50) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def train_one(model: LetterGPT, train_text: str, tokens: int, context: int = 64,
              batch: int = 64, lr: float = 3e-3, device: str = "cpu",
              seed: int = 0, log_every: int = 200) -> LetterGPT:
    torch.manual_seed(seed)
    model.to(device).train()
    ds = CharSeq(train_text[:tokens + context + 1], context)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    # one epoch plan: steps = len(dl); each training planned for its own length (Chinchilla-correct)
    total = max(len(dl), 1)
    step = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, total, lr)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    return model


def halvings_needed(gap0: float, frac: float = 0.5, per4x: float = 0.75) -> float:
    """How many x size multiplier to cut gap to frac? per4x = gap kept per 4x size."""
    import math
    # gap(N) = gap0 * (N/N0)^(-k), where 4^-k = per4x -> k = -log(per4x)/log(4)
    k = -math.log(per4x) / math.log(4.0)
    return math.exp(math.log(frac) / (-k))
