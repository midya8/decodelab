"""Inspectable temperature -> top-k -> top-p sampling, using float64 arithmetic.

Top-p is the smallest lexically tie-broken prefix reaching the threshold in
its *current*, normalized top-k distribution. This explicit composition is
not a claim that every generation library uses the same tie/cutoff rules.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .decode import DecodingConfig


def sampling_trace(dist: dict[str, float], config: DecodingConfig) -> dict:
    if not isinstance(dist, dict) or not dist:
        raise ValueError('distribution must be a nonempty mapping')
    if any(not isinstance(t, str) or not t for t in dist):
        raise ValueError('tokens must be nonempty strings')
    if any(isinstance(p, bool) or not isinstance(p, (float, int)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in dist.values()):
        raise ValueError('probabilities must be finite and nonnegative')
    total = math.fsum(dist.values())
    if abs(total - 1) > 1e-10:
        raise ValueError('probabilities must sum to one within 1e-10')
    items = sorted(((t, p / total) for t, p in dist.items() if p > 0),
                   key=lambda x: (-x[1], x[0]))
    # Subtract before division: even tiny positive temperatures leave at
    # least one exponent exactly zero. Tiny tail values may underflow to 0.
    maximum = math.log(items[0][1])
    powers = [(t, math.exp((math.log(p) - maximum) / config.temperature))
              for t, p in items]
    z = math.fsum(p for _, p in powers)
    tempered = [(t, p / z) for t, p in powers]
    selected = tempered[:config.top_k] if config.top_k is not None else tempered
    kmass = math.fsum(p for _, p in selected)
    knormal = [(t, p / kmass) for t, p in selected]
    kept = []
    cumulative = 0.0
    for t, p in knormal:
        if config.top_p is not None and config.top_p < 1 and kept and cumulative >= config.top_p:
            break
        kept.append((t, p))
        cumulative = math.fsum(p for _, p in kept)
    pmass = math.fsum(p for _, p in kept)
    output = {t: p / pmass for t, p in kept if p > 0}
    return {'base': dict(items), 'temperature': dict(tempered),
            'top_k': dict(knormal), 'output': output,
            'top_k_retained_temperature_mass': kmass,
            'nucleus_retained_top_k_mass': pmass,
            'nucleus_prefix_mass_before_last': math.fsum(p for _, p in kept[:-1]),
            'order': 'temperature -> normalized top-k -> normalized top-p',
            'ties': 'descending original probability, then lexical token'}
