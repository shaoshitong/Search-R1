"""Stable question selection; G5 payloads remain the caller's original objects.

No answer or EM is used by ``select_top_prefix``. Outcome selection happens
only after those same five trajectories have finished. ``payload`` can be the
rollout object itself or a remote artifact reference; selection never tokenizes,
copies, or regenerates it.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Candidate:
    question_id: str
    ordinal: int
    score: float
    strict_em: Optional[Tuple[int, ...]] = None
    payload: Any = None

    def __post_init__(self):
        if not isfinite(self.score):
            raise ValueError("Probe score must be finite")
        if self.strict_em is not None:
            values = tuple(self.strict_em)
            if len(values) != 5 or any(value not in (0, 1) for value in values):
                raise ValueError("strict_em requires the five original binary EM labels")
            object.__setattr__(self, "strict_em", values)

    @property
    def outcome(self):
        if self.strict_em is None:
            raise ValueError("This original G5 has not finished")
        correct = sum(self.strict_em)
        return "all_wrong" if correct == 0 else "all_correct" if correct == 5 else "mixed"


def _unique(candidates: Iterable[Candidate]):
    rows = list(candidates)
    if len({row.question_id for row in rows}) != len(rows):
        raise ValueError("A candidate pool must contain different questions")
    if len({row.ordinal for row in rows}) != len(rows):
        raise ValueError("Original pool ordinals must be unique")
    return rows


def score_order(candidate: Candidate):
    return -candidate.score, candidate.ordinal


def select_top_prefix(candidates: Iterable[Candidate], k: int = 1024):
    """Take score-descending questions, without consulting strict EM."""
    rows = _unique(candidates)
    if not 0 <= k <= len(rows):
        raise ValueError("Prefix selection exceeds its candidate pool")
    return sorted(rows, key=score_order)[:k]


def select_actual_mixed(candidates: Iterable[Candidate], k: int = 512, *, allow_all_wrong=True):
    """Mixed first; all-correct and all-wrong form one score-ranked fill pool.

    This applies to both ordinary Top1024 and all2560 periodic full rollouts.
    The caller retains the entire periodic pool for the later probe fit.
    """
    rows = _unique(candidates)
    if not allow_all_wrong:
        rows = [row for row in rows if row.outcome != "all_wrong"]
    if not 0 <= k <= len(rows):
        raise ValueError("Actor selection exceeds its eligible completed pool")
    return sorted(rows, key=lambda row: (row.outcome != "mixed", *score_order(row)))[:k]


def group_counts(candidates: Iterable[Candidate]):
    counts = dict(all_correct=0, all_wrong=0, mixed=0)
    total = 0
    for candidate in candidates:
        counts[candidate.outcome] += 1
        total += 1
    return dict(questions=total, **counts,
                mixed_fraction=counts["mixed"] / total if total else 0.0)


def candidate_budget(outer: int):
    """outer200 has no unused final probe fit and uses the ordinary budget."""
    return 2560 if 0 < outer < 200 and outer % 10 == 0 else 5120
