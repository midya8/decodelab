"""DecodeLab: decoding strategies for language models, from scratch."""

from .model import NGramModel
from .decode import greedy, sample, beam_search, beam_trace, DecodingConfig
from .filtering import sampling_trace
from .paths import enumerate_paths
from .metrics import distinct_n, repetition_rate, token_entropy

__version__ = "0.2.0"

__all__ = [
    "NGramModel",
    "greedy",
    "sample",
    "beam_search",
    "beam_trace",
    "sampling_trace",
    "enumerate_paths",
    "DecodingConfig",
    "distinct_n",
    "repetition_rate",
    "token_entropy",
]
