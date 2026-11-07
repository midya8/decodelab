"""DecodeLab: decoding strategies for language models, from scratch."""

from .model import NGramModel
from .decode import greedy, sample, beam_search, DecodingConfig
from .metrics import distinct_n, repetition_rate, token_entropy

__version__ = "0.1.0"

__all__ = [
    "NGramModel",
    "greedy",
    "sample",
    "beam_search",
    "DecodingConfig",
    "distinct_n",
    "repetition_rate",
    "token_entropy",
]
