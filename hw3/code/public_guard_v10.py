#!/usr/bin/env python3
"""Discard train candidates that copy HW3 PUBLIC user text.

PUBLIC is used only to build an in-memory rejection index.  This module never
returns or writes PUBLIC content, schemas, labels, or model responses.
"""

from __future__ import annotations

import collections
import difflib
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def word_ngrams(text: str, n: int = 4) -> set[str]:
    words = normalized(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index:index + n]) for index in range(len(words) - n + 1)}


class PublicOverlapGuard:
    """Exact and conservative near-copy rejection for user messages."""

    def __init__(self, paths: Iterable[Path], threshold: float = 0.92) -> None:
        self.threshold = threshold
        texts: set[str] = set()
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    for message in row.get("messages") or []:
                        if message.get("role") == "user" and isinstance(message.get("content"), str):
                            value = normalized(message["content"])
                            if value:
                                texts.add(value)

        self._texts = sorted(texts)
        self._exact = set(self._texts)
        self._grams = [word_ngrams(text) for text in self._texts]
        posting: dict[str, list[int]] = collections.defaultdict(list)
        for index, grams in enumerate(self._grams):
            for gram in grams:
                posting[gram].append(index)
        self._posting = posting
        self.checked = 0
        self.exact_hits = 0
        self.near_hits = 0

    def overlaps(self, text: str) -> bool:
        value = normalized(text)
        if not value:
            return False
        self.checked += 1
        if value in self._exact:
            self.exact_hits += 1
            return True

        # Short fragments share too many accidental n-grams; exact matching is
        # the safer rejection rule for them.
        if len(value.split()) < 8:
            return False
        grams = word_ngrams(value)
        rare_shared = sorted(
            (gram for gram in grams if gram in self._posting),
            key=lambda gram: len(self._posting[gram]),
        )[:8]
        candidates: set[int] = set()
        for gram in rare_shared:
            if len(self._posting[gram]) <= 100:
                candidates.update(self._posting[gram])
        for index in candidates:
            other = self._grams[index]
            union = len(grams | other)
            jaccard = len(grams & other) / union if union else 0.0
            if jaccard >= self.threshold or difflib.SequenceMatcher(
                None, value, self._texts[index]
            ).ratio() >= 0.95:
                self.near_hits += 1
                return True
        return False

    def row_overlaps(self, messages: list[dict[str, Any]]) -> bool:
        return any(
            self.overlaps(message.get("content", ""))
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        )

    def stats(self) -> dict[str, int | float]:
        return {
            "public_guard_checked_user_messages": self.checked,
            "public_guard_exact_hits": self.exact_hits,
            "public_guard_near_hits": self.near_hits,
            "public_guard_4gram_jaccard_threshold": self.threshold,
        }
