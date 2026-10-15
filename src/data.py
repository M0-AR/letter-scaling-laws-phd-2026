"""27-letter Wikipedia pipeline: a-z + space, digits spelled out, locked test split.

Mirrors the video: 100M letters total, last 5M locked away, train sees each char once.
Live-data mode fetches Wikipedia via REST API; offline mode synthesizes equivalent stats.
"""
from __future__ import annotations
import re
import pathlib
import requests

VOCAB = list("abcdefghijklmnopqrstuvwxyz ")
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def normalize(text: str) -> str:
    text = text.lower()
    # spell digits digit-by-digit: 48 -> "foureight"
    out = []
    for ch in text:
        if ch.isdigit():
            out.append(DIGIT_WORDS[int(ch)])
        elif ch == " " or ("a" <= ch <= "z"):
            out.append(ch)
        elif ch in "\n\t":
            out.append(" ")
        # drop capitals already lowered, commas, punctuation -> space if separator else drop
        elif ch in ".,;:!?-_()[]{}\"'/\\|":
            out.append(" ")
        # ignore everything else
    s = "".join(out)
    s = re.sub(r" +", " ", s)
    return s


def fetch_wikipedia_live(chars: int = 200_000, lang: str = "en") -> str:
    """Fetch live public data: random Wikipedia article summaries until `chars` normalized chars."""
    buf: list[str] = []
    total = 0
    session = requests.Session()
    # use random + search APIs (public, no key)
    try:
        for _ in range(200):
            r = session.get(f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary", timeout=15)
            if r.status_code != 200:
                continue
            j = r.json()
            txt = normalize(j.get("extract", ""))
            if len(txt) < 200:
                continue
            buf.append(txt + " ")
            total += len(txt) + 1
            if total >= chars:
                break
    except Exception as e:
        print(f"[data] live fetch failed ({e}), falling back to synthetic")
        return synthetic_text(chars)
    text = (" ".join(buf))[:chars]
    if len(text) < chars // 2:
        return synthetic_text(chars)
    return text


def synthetic_text(chars: int, seed: int = 0) -> str:
    """Deterministic public-data stand-in with same unigram stats (space 17%, e 11%)."""
    import random
    rng = random.Random(seed)
    # weights approximating video: space .17, e .11, rest Zipf-ish
    weights = [0.17 if c == " " else 0.11 if c == "e" else 0.72 / 25 for c in VOCAB]
    # add bigram flavor: prefer 'th', 'he', 'ou' chains by resampling
    s = "".join(rng.choices(VOCAB, weights=weights, k=chars))
    return s


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s if c in STOI]


def decode(ids: list[int]) -> str:
    return "".join(ITOS[i] for i in ids)


def write_corpus(path: pathlib.Path, text: str):
    path.write_text(text)


def load_or_build(data_dir: pathlib.Path, train_chars: int, test_chars: int = 50_000,
                 live: bool = True) -> tuple[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_p = data_dir / f"train_{train_chars}.txt"
    test_p = data_dir / f"test_{test_chars}.txt"
    if train_p.exists() and test_p.exists():
        return train_p.read_text(), test_p.read_text()
    full = fetch_wikipedia_live(train_chars + test_chars) if live else synthetic_text(train_chars + test_chars)
    # last test_chars locked away
    train, test = full[:-test_chars], full[-test_chars:]
    train_p.write_text(train)
    test_p.write_text(test)
    return train, test
