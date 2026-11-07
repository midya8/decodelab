"""Decoding strategies: greedy, temperature/top-k/top-p sampling, beam search.

All strategies consume the model's exact next-token distribution. Sampling
takes an explicit ``random.Random`` so runs are reproducible; beam search is
fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from .model import NGramModel


@dataclass(frozen=True)
class DecodingConfig:
    max_tokens: int = 20
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def _filtered(dist: dict[str, float], config: DecodingConfig) -> dict[str, float]:
    """Apply temperature, top-k and top-p, then renormalize."""
    items = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    if config.temperature != 1.0:
        items = [(t, p ** (1.0 / config.temperature)) for t, p in items]
    if config.top_k is not None:
        items = items[: config.top_k]
    if config.top_p is not None:
        kept: list[tuple[str, float]] = []
        cumulative = 0.0
        for tok, p in items:
            if kept and cumulative >= config.top_p:
                break
            kept.append((tok, p))
            cumulative += p
        items = kept
    total = sum(p for _, p in items)
    return {t: p / total for t, p in items}


def greedy(model: NGramModel, prompt: list[str], config: DecodingConfig | None = None) -> list[str]:
    """Argmax at every step; ties break on higher corpus frequency, then the
    lexicographically smaller token. Fully deterministic."""
    config = config or DecodingConfig()
    tokens = list(prompt)
    for _ in range(config.max_tokens):
        dist = model.distribution(tokens)
        nxt = min(dist, key=lambda t: (-dist[t], -model.unigram_count(t), t))
        if nxt == NGramModel.END:
            break
        tokens.append(nxt)
    return tokens


def sample(
    model: NGramModel,
    prompt: list[str],
    config: DecodingConfig | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample from the temperature/top-k/top-p filtered distribution."""
    config = config or DecodingConfig()
    rng = rng or random.Random(0)
    tokens = list(prompt)
    for _ in range(config.max_tokens):
        dist = _filtered(model.distribution(tokens), config)
        items = sorted(dist.items())
        draw = rng.random()
        cumulative = 0.0
        nxt = items[-1][0]
        for tok, p in items:
            cumulative += p
            if draw < cumulative:
                nxt = tok
                break
        if nxt == NGramModel.END:
            break
        tokens.append(nxt)
    return tokens


def beam_search(
    model: NGramModel,
    prompt: list[str],
    beam_width: int,
    config: DecodingConfig | None = None,
) -> list[list[str]]:
    """Beam search by total log probability. Returns finished + live beams,
    best first; each beam includes the prompt."""
    import math

    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    config = config or DecodingConfig()
    # (tokens, logprob, finished)
    beams: list[tuple[list[str], float, bool]] = [(list(prompt), 0.0, False)]
    for _ in range(config.max_tokens):
        candidates: list[tuple[list[str], float, bool]] = []
        for tokens, logprob, finished in beams:
            if finished:
                candidates.append((tokens, logprob, True))
                continue
            dist = model.distribution(tokens)
            for tok, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[: max(beam_width * 2, 4)]:
                if tok == NGramModel.END:
                    candidates.append((tokens, logprob + math.log(p), True))
                else:
                    candidates.append((tokens + [tok], logprob + math.log(p), False))
        candidates.sort(key=lambda b: (-b[1], b[0]))
        beams = candidates[:beam_width]
        if all(f for _, _, f in beams):
            break
    return [tokens for tokens, _, _ in beams]
