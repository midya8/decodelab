"""Add-k smoothed n-gram language model over pre-tokenized text.

The model is deliberately simple: it exists to give decoding strategies a
real, reproducible next-token distribution to work against, fully offline.
Counts come from a corpus; the distribution is exact, never sampled.
"""

from __future__ import annotations

import math
from collections import defaultdict


class NGramModel:
    """N-gram model with add-k smoothing over a fixed vocabulary.

    ``order`` is the n in n-gram (order 2 = bigram). Contexts longer than
    ``order - 1`` are truncated to the most recent tokens; short contexts
    at sequence start are padded with ``<s>``. The end of a sequence is the
    ``</s>`` token, which can be produced but never appears in a context.
    """

    START = "<s>"
    END = "</s>"

    def __init__(self, order: int, k: float = 0.1) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if k <= 0:
            raise ValueError("k must be > 0")
        self.order = order
        self.k = k
        self._counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._vocab: set[str] = set()
        self._unigram: dict[str, int] = defaultdict(int)

    def train(self, sentences: list[list[str]]) -> None:
        """Count n-grams from pre-tokenized sentences. Replaces prior counts."""
        self._counts = defaultdict(lambda: defaultdict(int))
        self._vocab = set()
        self._unigram = defaultdict(int)
        for sent in sentences:
            if not sent:
                continue
            self._vocab.update(sent)
            for tok in sent:
                self._unigram[tok] += 1
            padded = [self.START] * (self.order - 1) + list(sent) + [self.END]
            for i in range(self.order - 1, len(padded)):
                context = tuple(padded[max(0, i - (self.order - 1)):i])
                self._counts[context][padded[i]] += 1

    def unigram_count(self, token: str) -> int:
        """Corpus frequency of a token, used for deterministic tie-breaking."""
        return self._unigram.get(token, 0)

    @property
    def vocab(self) -> frozenset[str]:
        return frozenset(self._vocab)

    def distribution(self, context: tuple[str, ...] | list[str]) -> dict[str, float]:
        """Exact next-token distribution over vocab + END for a context.

        Every token in the vocabulary gets add-k probability mass, so the
        distribution is defined for unseen contexts. Returns a dict mapping
        token to probability, summing to 1 (within float tolerance).
        """
        if not self._vocab:
            raise ValueError("model is not trained")
        context = tuple(context)[-(self.order - 1):] if self.order > 1 else ()
        counts = self._counts.get(context, {})
        total = sum(counts.values()) + self.k * (len(self._vocab) + 1)
        dist = {tok: (counts.get(tok, 0) + self.k) / total for tok in self._vocab}
        dist[self.END] = (counts.get(self.END, 0) + self.k) / total
        return dist

    def log_probability(self, sentence: list[str]) -> float:
        """Sum of log probabilities of each token in context, plus END."""
        if not sentence:
            raise ValueError("sentence must be non-empty")
        padded = [self.START] * (self.order - 1) + list(sentence) + [self.END]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            context = padded[max(0, i - (self.order - 1)):i]
            prob = self.distribution(context)[padded[i]]
            total += math.log(prob)
        return total
