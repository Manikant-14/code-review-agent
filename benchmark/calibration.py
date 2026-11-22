"""Calibration and ranking of a confidence score against true-positive labels (docs/evaluation.md).

Works on any score: reviewer self-confidence now, critic or gate confidence later. Every finding with
a score counts (positive AND negative runs); the label is the finding's true-positive status.

  Brier        mean squared error of the score as a probability; compared with the base-rate
               predictor (always say the overall TP rate) as a skill score, 1 = perfect, 0 = no
               better than the base rate, negative = worse.
  ECE          expected calibration error over up to k equal-mass bins (ties kept together, because
               scores cluster on a few values), the weighted gap between mean score and observed rate.
  AUROC        chance that a random true positive scores above a random false positive (ties 0.5):
               ranking quality alone, which is what a threshold gate depends on.

Intervals: cluster bootstrap over cases, as everywhere else.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from benchmark import metrics as mt
from benchmark.labeling import Labeled, Run

KEYS = ("brier", "brier_skill", "ece", "auroc")


def pairs_of(findings: list[Labeled]) -> list[tuple[float, int]]:
    """(score, 1 if true positive else 0) for findings that have a score; pending labels raise."""
    mt._check_labels(findings)
    return [(f.score, 1 if f.is_tp else 0) for f in findings if f.score is not None]


def brier(pairs) -> Optional[float]:
    return sum((s - y) ** 2 for s, y in pairs) / len(pairs) if pairs else None


def brier_reference(pairs) -> Optional[float]:
    """Brier score of always predicting the overall true-positive rate: p * (1 - p)."""
    if not pairs:
        return None
    p = sum(y for _, y in pairs) / len(pairs)
    return p * (1 - p)


def brier_skill(pairs) -> Optional[float]:
    ref = brier_reference(pairs)
    return None if not ref else 1 - brier(pairs) / ref


def reliability(pairs, k: int = 5) -> list[dict]:
    """Up to k equal-mass bins by score; identical scores never straddle two bins."""
    if not pairs:
        return []
    ordered = sorted(pairs)
    target = len(ordered) / k
    groups, i = [], 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        groups.append(ordered[i:j])
        i = j
    bins, cur = [], []
    for g in groups:
        cur += g
        if len(cur) >= target and len(bins) < k - 1:
            bins.append(cur)
            cur = []
    if cur:
        bins.append(cur)
    return [{"n": len(b), "lo": b[0][0], "hi": b[-1][0],
             "mean_score": sum(s for s, _ in b) / len(b), "rate": sum(y for _, y in b) / len(b)} for b in bins]


def ece(pairs, k: int = 5) -> Optional[float]:
    if not pairs:
        return None
    return sum(b["n"] / len(pairs) * abs(b["rate"] - b["mean_score"]) for b in reliability(pairs, k))


def auroc(pairs) -> Optional[float]:
    """Mann-Whitney AUC from tie-averaged ranks; None unless both classes are present."""
    pos = sum(y for _, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(pairs)
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2                       # ranks are 1-based: i+1 .. j
        rank_sum += avg_rank * sum(y for _, y in ordered[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def calibration_metrics(findings: list[Labeled], k: int = 5) -> dict:
    p = pairs_of(findings)
    return {"n": len(p), "base_rate": (sum(y for _, y in p) / len(p)) if p else None,
            "brier": brier(p), "brier_reference": brier_reference(p), "brier_skill": brier_skill(p),
            "ece": ece(p, k), "auroc": auroc(p), "bins": reliability(p, k)}


def calibration_summary(runs: list[Run], findings: list[Labeled], B: int = 2000, seed: int = 0, k: int = 5) -> dict:
    """Point estimates plus 95% cluster-bootstrap intervals (lo, hi, replicates where defined)."""
    point = calibration_metrics(findings, k)
    reps = defaultdict(list)
    for _, fs in mt.bootstrap_samples(runs, findings, B, seed):
        m = calibration_metrics(fs, k)
        for key in KEYS:
            reps[key].append(m[key])
    return {**point, "ci": {key: mt._ci(reps[key]) for key in KEYS}}