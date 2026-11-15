"""Exact finite-cap stopping laws for DecodeLab's supplied integer tables.

This is an unfiltered count-ratio model, not the floating sampling pipeline.
No continuation strings are enumerated and no tail beyond the cap is inferred.
"""
from fractions import Fraction as F
from .audit import TableModel


def stopping_law(counts=None, prompt=None, cap=20):
    """Full rational state-flow and generated-length law, including censored mass.

    EOS consumes a decision step but emits no ordinary token. All non-EOS mass
    after the final decision is a cap event, with no additional EOS factor.
    """
    model = TableModel(counts)
    if type(cap) is not int or not 1 <= cap <= 128:
        raise ValueError('cap must be an integer in [1, 128]')
    prompt = [] if prompt is None else prompt
    if type(prompt) is not list or any(type(t) is not str or t not in ('a', 'b') for t in prompt):
        raise ValueError('prompt must be a list of a/b tokens')
    frontier = {prompt[-1] if prompt else 'start': F(1)}
    length_mass = [F(0)] * (cap + 1)
    steps = []
    for depth in range(cap):
        transitions = []
        next_mass = {'a': F(0), 'b': F(0)}
        stopped = F(0)
        incoming = sum(frontier.values(), F(0))
        for state, mass in sorted(frontier.items()):
            row = model.counts[state]
            total = sum(row)
            for token, weight in zip(('a', 'b', '</s>'), row):
                probability = F(weight, total)
                flow = mass * probability
                transitions.append({'state': state, 'token': token,
                                    'probability': str(probability), 'mass': str(flow)})
                if token == '</s>':
                    stopped += flow
                else:
                    next_mass[token] += flow
        length_mass[depth] = stopped
        steps.append({'generated_before': depth, 'incoming': {k:str(v) for k,v in sorted(frontier.items())},
                      'incoming_mass': str(incoming), 'transitions': transitions,
                      'eos_mass': str(stopped), 'surviving_mass': str(sum(next_mass.values(), F(0))),
                      'next_states': {k:str(v) for k,v in sorted(next_mass.items())}})
        frontier = next_mass
    capped = sum(frontier.values(), F(0))
    length_mass[cap] = capped
    expected = sum((i*p for i,p in enumerate(length_mass)), F(0))
    second = sum((i*i*p for i,p in enumerate(length_mass)), F(0))
    return {'schema': 'decodelab-stopping-v1', 'counts': model.counts, 'prompt': list(prompt), 'cap': cap,
            'steps': steps, 'length_probability': [str(p) for p in length_mass],
            'eos_probability': str(1-capped), 'cap_probability': str(capped),
            'expected_generated_tokens': str(expected), 'variance_generated_tokens': str(second-expected*expected),
            'scope': 'exact supplied integer-weight table ratios; unfiltered, prompt-conditioned, cap-censored; not floating sampling, beam search, an infinite-horizon tail or language quality'}
