"""Deterministic whitespace/punctuation tokenization for corpora and prompts."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9']+|[^\s\w]")


def tokenize(text: str) -> list[str]:
    """Lowercase word/punctuation tokens, in order."""
    return _TOKEN.findall(text.lower())


def tokenize_sentences(text: str) -> list[list[str]]:
    """Split text into sentences on ., !, ? and newlines; each sentence is a
    token list with the terminal punctuation kept as its own token."""
    sentences: list[list[str]] = []
    current: list[str] = []
    for tok in tokenize(text):
        current.append(tok)
        if tok in (".", "!", "?"):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences
