"""Generation-quality metrics: distinct-n, repetition rate, token entropy."""

from __future__ import annotations

import math
from collections import Counter


def distinct_n(tokens: list[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams. 1.0 means no n-gram repeats;
    0.0 for a sequence shorter than n (defined, by convention, as no
    diversity evidence)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(tokens: list[str]) -> float:
    """Fraction of tokens identical to their immediate predecessor."""
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return repeats / (len(tokens) - 1)


def token_entropy(tokens: list[str]) -> float:
    """Shannon entropy in bits of the unigram distribution of the tokens."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def evaluate(tokens: list[str]) -> dict[str, float]:
    """All metrics for one generated sequence."""
    return {
        "tokens": float(len(tokens)),
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "repetition_rate": repetition_rate(tokens),
        "entropy_bits": token_entropy(tokens),
    }
