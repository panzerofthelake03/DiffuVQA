"""Minimal answer preprocessing utilities.

This module provides a small, dependency-free implementation of
`find_most_similar_answers` used by the repository. It uses
difflib.SequenceMatcher to compute string similarity and returns the
top-k most similar candidate answers.

If you have a more sophisticated answer normalization / matching
utility, replace this file with the original implementation.
"""
from typing import List, Tuple
from difflib import SequenceMatcher

__all__ = ["find_most_similar_answers"]


def _normalize(s: str) -> str:
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())


def find_most_similar_answers(target: str, candidates: List[str], top_k: int = 1) -> List[Tuple[str, float]]:
    """Find the most similar answers to `target` among `candidates`.

    Args:
        target: target answer string.
        candidates: list of candidate answer strings.
        top_k: number of top matches to return.

    Returns:
        A list of (candidate, score) tuples sorted by descending score.
        Score is in [0.0, 1.0] where 1.0 means exact match.

    Note: This is intentionally lightweight. Replace with a dataset-specific
    implementation if you need token-level matching or embedding-based similarity.
    """
    if not isinstance(candidates, (list, tuple)):
        # try to coerce common container types to list
        try:
            candidates = list(candidates)
        except Exception:
            candidates = [str(candidates)]

    t = _normalize(target)
    scores = []
    for c in candidates:
        c_norm = _normalize(c)
        if not t and not c_norm:
            score = 1.0
        else:
            score = SequenceMatcher(None, t, c_norm).ratio()
        scores.append((c, float(score)))

    # sort by score desc
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max(1, top_k)]
