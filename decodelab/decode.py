"""Decoding strategies: greedy, temperature/top-k/top-p sampling, beam search.

All strategies consume the model's exact next-token distribution. Sampling
takes an explicit ``random.Random`` so runs are reproducible; beam search is
fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import math

from .filtering import sampling_trace

from .model import NGramModel


@dataclass(frozen=True)
class DecodingConfig:
    max_tokens: int = 20
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens must be an integer >= 1")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be an integer >= 1")
        if self.top_p is not None and (isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)) or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError("top_p must be finite and in (0, 1]")


def _filtered(dist: dict[str, float], config: DecodingConfig) -> dict[str, float]:
    """Stable staged sampling; see sampling_trace for all intermediate values."""
    return sampling_trace(dist, config)["output"]


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


def beam_trace(model: NGramModel, prompt: list[str], beam_width: int,
               config: DecodingConfig | None = None, *, record: bool = True) -> dict:
    """Full expansion/pruning witnesses for unfiltered model log probability.

    Finished EOS hypotheses stay in the beam. Cap-censored hypotheses have no
    EOS factor. Temperature/top-k/top-p are sampling-only and ignored here.
    """
    if type(beam_width) is not int or beam_width < 1:
        raise ValueError("beam_width must be an integer >= 1")
    config = config or DecodingConfig()
    beams = [{"tokens": list(prompt), "log_probability": 0.0, "finished": False}]
    history = []
    expanded = 0
    for step in range(config.max_tokens):
        candidates = []
        for parent, beam in enumerate(beams):
            if beam["finished"]:
                candidates.append(dict(beam, parent=parent, emitted=None))
                continue
            items = sorted(model.distribution(beam["tokens"]).items(), key=lambda x: (-x[1], x[0]))
            if not record:
                items = items[:max(beam_width * 2, 4)]
            for token, p in items:
                if p <= 0:
                    continue
                candidates.append({"tokens": beam["tokens"] + ([] if token == NGramModel.END else [token]),
                                   "log_probability": beam["log_probability"] + math.log(p),
                                   "finished": token == NGramModel.END,
                                   "parent": parent, "emitted": token})
        expanded += len(candidates)
        if record and expanded > 100000:
            raise ValueError("beam witness exceeds 100000 candidate records")
        candidates.sort(key=lambda b: (-b["log_probability"], b["tokens"], not b["finished"]))
        beams = candidates[:beam_width]
        if record:
            history.append({"step": step+1, "candidates": candidates, "survivors": beams})
        if all(b["finished"] for b in beams):
            break
    return {"beams": beams, "steps": history,
            "objective": "unfiltered conditional continuation log probability; EOS scored, cap censored"}


def beam_search(model: NGramModel, prompt: list[str], beam_width: int,
                config: DecodingConfig | None = None) -> list[list[str]]:
    """Compatibility output. Use beam_trace to distinguish EOS from the cap."""
    return [b["tokens"] for b in beam_trace(model, prompt, beam_width, config, record=False)["beams"]]
