"""Bounded exhaustive continuation trees. Exponential, not an LM decoder.

Each outcome is either EOS-terminated (EOS is scored, not displayed) or
right-censored by the generation cap (no invented terminal EOS factor).
The prompt is conditioned on, never included in the continuation score.
"""
from __future__ import annotations
import math
from .decode import DecodingConfig, _filtered
from .model import NGramModel


def enumerate_paths(model, prompt: list[str], config: DecodingConfig,
                    max_nodes: int = 100000) -> dict:
    if type(max_nodes) is not int or not 1 <= max_nodes <= 1000000:
        raise ValueError('max_nodes must be an integer in [1, 1000000]')
    if not isinstance(prompt, list) or any(not isinstance(x, str) or x in (NGramModel.START, NGramModel.END) for x in prompt):
        raise ValueError('prompt must contain ordinary string tokens')
    if config.max_tokens > 12:
        raise ValueError("exhaustive trees require max_tokens <= 12")
    outcomes = []
    visited = 0
    stack = [(list(prompt), [], 1.0, 0.0, 0)]
    while stack:
        context, generated, mass, raw_log, depth = stack.pop()
        visited += 1
        if visited > max_nodes:
            raise ValueError('exhaustive tree exceeds max_nodes; no partial certificate returned')
        if depth == config.max_tokens:
            outcomes.append({'tokens': generated, 'stop': 'cap', 'probability': mass,
                             'model_log_probability': raw_log})
            continue
        base = model.distribution(context)
        if len(base) > 32:
            raise ValueError("exhaustive trees support at most 32 next-token entries")
        policy = _filtered(base, config)
        for token, q in sorted(policy.items(), reverse=True):
            logp = raw_log + math.log(base[token])
            if token == NGramModel.END:
                visited += 1
                if visited > max_nodes:
                    raise ValueError('exhaustive tree exceeds max_nodes; no partial certificate returned')
                outcomes.append({'tokens': generated, 'stop': 'eos', 'probability': mass*q,
                                 'model_log_probability': logp})
            else:
                stack.append((context+[token], generated+[token], mass*q, logp, depth+1))
    outcomes.sort(key=lambda x: (-x['probability'], x['tokens'], x['stop']))
    return {'outcomes': outcomes, 'visited_nodes': visited,
            'total_probability': math.fsum(x['probability'] for x in outcomes),
            'eos_probability': math.fsum(x['probability'] for x in outcomes if x['stop']=='eos'),
            'cap_probability': math.fsum(x['probability'] for x in outcomes if x['stop']=='cap'),
            'expected_generated_tokens': math.fsum(len(x['tokens'])*x['probability'] for x in outcomes)}
