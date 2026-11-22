"""Metrics, intervals and the operating curve (docs/evaluation.md). Pure functions, no I/O.

A metric with nothing to measure is None (printed n/a), never 0 or 1: e.g. auto-post precision when
nothing was auto-posted. Proportions over findings get Wilson intervals; run-level metrics and the
curve get a cluster bootstrap over CASES (the positive and negative runs of one bug stay together).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import replace
from typing import Callable, Iterator, Optional

from benchmark.labeling import Labeled, Run
from src import config

AUTO, ESC = "auto_post", "escalate"
RUN_LEVEL = ("detection_recall", "autopost_recall", "fp_load_auto", "fp_load_escalated", "fix_flag_rate")
CURVE_KEYS = ("auto_post_precision", "escalation_rate", "autopost_recall", "detection_recall")


# ---- intervals -------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    if n <= 0:
        return None
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(sorted_vals: list[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _ci(values: list[Optional[float]]) -> Optional[tuple[float, float, int]]:
    """95% percentile interval over the replicates where the metric was defined, plus how many."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return percentile(vals, 0.025), percentile(vals, 0.975), len(vals)


# ---- policies: which findings are auto-posted ------------------------------------------------
def all_auto(f: Labeled, _param=None) -> bool:
    """No gate: everything is posted."""
    return True


def score_at_least(f: Labeled, t: float) -> bool:
    """Baseline: auto-post when the score (e.g. reviewer self-confidence) reaches t."""
    return f.score is not None and f.score >= t


def severity_scaled(f: Labeled, s: float) -> bool:
    """The gate's per-severity thresholds (config.AUTO_POST_THRESHOLDS) times a scale s.

    Applies the rule-based severity floor (security is always HIGH). No score -> escalate.
    """
    if f.score is None:
        return False
    sev = (config.Severity.HIGH if config.Category(f.category) in config.HIGH_SEVERITY_CATEGORIES
           else config.Severity(f.severity))
    return f.score >= min(1.0, config.AUTO_POST_THRESHOLDS[sev] * s)


POLICIES = {"all_auto": all_auto, "score": score_at_least, "severity_scaled": severity_scaled}


def with_decisions(findings: list[Labeled], auto_fn: Callable, param=None) -> list[Labeled]:
    return [replace(f, decision=AUTO if auto_fn(f, param) else ESC) for f in findings]


# ---- metrics ---------------------------------------------------------------------------------
def _ratio(a: int, b: int) -> Optional[float]:
    return None if b == 0 else a / b


def _check_labels(findings: list[Labeled]) -> None:
    for f in findings:
        if f.is_tp is None:
            raise ValueError(f"finding {f.id} is still waiting for the judge")


def _metrics(runs: list[Run], findings: list[Labeled], auto: list[bool]) -> dict:
    pos_runs = {r.run_id for r in runs if r.kind == "positive"}
    neg_runs = {r.run_id for r in runs if r.kind == "negative"}
    n = len(findings)
    k_auto = sum(auto)
    k_auto_tp = sum(1 for f, a in zip(findings, auto) if a and f.is_tp)
    pos_n = sum(1 for f in findings if f.kind == "positive")
    neg_n = n - pos_n
    k_esc_pos = sum(1 for f, a in zip(findings, auto) if not a and f.kind == "positive")
    k_esc_neg = sum(1 for f, a in zip(findings, auto) if not a and f.kind == "negative")
    tp_runs = {f.run_id for f in findings if f.is_tp} & pos_runs
    auto_tp_runs = {f.run_id for f, a in zip(findings, auto) if a and f.is_tp} & pos_runs
    neg_auto = sum(1 for f, a in zip(findings, auto) if a and f.kind == "negative")
    return {
        "n_pos_runs": len(pos_runs), "n_neg_runs": len(neg_runs), "n_findings": n,
        "n_auto": k_auto, "n_escalated": n - k_auto, "n_auto_tp": k_auto_tp,
        "n_pos_findings": pos_n, "n_neg_findings": neg_n,
        "n_esc_pos": k_esc_pos, "n_esc_neg": k_esc_neg,
        "auto_post_precision": _ratio(k_auto_tp, k_auto),
        "escalation_rate": _ratio(n - k_auto, n),
        "escalation_rate_pos": _ratio(k_esc_pos, pos_n),
        "escalation_rate_neg": _ratio(k_esc_neg, neg_n),
        "detection_recall": _ratio(len(tp_runs), len(pos_runs)),
        "autopost_recall": _ratio(len(auto_tp_runs), len(pos_runs)),
        "fp_load_auto": _ratio(neg_auto, len(neg_runs)),
        "fp_load_escalated": _ratio(k_esc_neg, len(neg_runs)),
        "fix_flag_rate": _ratio(len({f.run_id for f in findings if f.kind == "negative"}), len(neg_runs)),
    }


def metrics(runs: list[Run], findings: list[Labeled]) -> dict:
    """Metrics for findings that already carry decisions (from a gate or with_decisions)."""
    _check_labels(findings)
    for f in findings:
        if f.decision not in (AUTO, ESC):
            raise ValueError(f"finding {f.id} has no decision: apply a policy or run the gate first")
    return _metrics(runs, findings, [f.decision == AUTO for f in findings])


# ---- cluster bootstrap over cases ------------------------------------------------------------
def bootstrap_samples(runs: list[Run], findings: list[Labeled], B: int,
                      seed: int = 0) -> Iterator[tuple[list[Run], list[Labeled]]]:
    """B resamples of CASES with replacement; a case keeps all its runs and findings together.

    Each drawn copy of a case gets its own run ids, so a case drawn twice counts twice.
    """
    rng = random.Random(seed)
    runs_by, finds_by = defaultdict(list), defaultdict(list)
    for r in runs:
        runs_by[r.case_id].append(r)
    for f in findings:
        finds_by[f.case_id].append(f)
    cases = sorted(runs_by)
    for _ in range(B):
        rs: list[Run] = []
        fs: list[Labeled] = []
        for j, c in enumerate(rng.choices(cases, k=len(cases))):
            tag = f"#b{j}"
            rs += [replace(r, run_id=r.run_id + tag) for r in runs_by[c]]
            fs += [replace(f, run_id=f.run_id + tag) for f in finds_by.get(c, [])]
        yield rs, fs


def cluster_bootstrap(runs: list[Run], findings: list[Labeled], names=RUN_LEVEL,
                      B: int = 2000, seed: int = 0) -> dict[str, Optional[tuple[float, float, int]]]:
    """name -> (lo, hi, replicates where defined). Findings must already carry decisions."""
    _check_labels(findings)
    reps = defaultdict(list)
    for rs, fs in bootstrap_samples(runs, findings, B, seed):
        m = _metrics(rs, fs, [f.decision == AUTO for f in fs])
        for k in names:
            reps[k].append(m[k])
    return {k: _ci(reps[k]) for k in names}


def summarize(runs: list[Run], findings: list[Labeled], B: int = 2000, seed: int = 0) -> dict:
    """Point estimates with intervals as the protocol asks: Wilson for proportions over findings,
    cluster bootstrap for run-level metrics. Each entry: {value, ci, method}."""
    m = metrics(runs, findings)
    boot = cluster_bootstrap(runs, findings, RUN_LEVEL, B, seed)
    out = {
        "auto_post_precision": {"value": m["auto_post_precision"], "method": "wilson",
                                "ci": wilson(m["n_auto_tp"], m["n_auto"])},
        "escalation_rate": {"value": m["escalation_rate"], "method": "wilson",
                            "ci": wilson(m["n_escalated"], m["n_findings"])},
        "escalation_rate_pos": {"value": m["escalation_rate_pos"], "method": "wilson",
                                "ci": wilson(m["n_esc_pos"], m["n_pos_findings"])},
        "escalation_rate_neg": {"value": m["escalation_rate_neg"], "method": "wilson",
                                "ci": wilson(m["n_esc_neg"], m["n_neg_findings"])},
    }
    for k in RUN_LEVEL:
        out[k] = {"value": m[k], "method": "bootstrap", "ci": boot[k][:2] if boot[k] else None,
                  "n_valid": boot[k][2] if boot[k] else 0}
    out["counts"] = {k: m[k] for k in ("n_pos_runs", "n_neg_runs", "n_findings", "n_auto", "n_escalated")}
    return out


def operating_curve(runs: list[Run], findings: list[Labeled], auto_fn: Callable, params: list,
                    B: int = 2000, seed: int = 0) -> list[dict]:
    """Auto-post precision vs escalation rate as the policy parameter is swept, with bootstrap bands.

    Ignores any decisions already on the findings: auto_fn(f, param) decides each point.
    """
    _check_labels(findings)

    def point(rs, fs, p):
        return _metrics(rs, fs, [auto_fn(f, p) for f in fs])

    base = [point(runs, findings, p) for p in params]
    reps = [defaultdict(list) for _ in params]
    for rs, fs in bootstrap_samples(runs, findings, B, seed):
        for i, p in enumerate(params):
            m = point(rs, fs, p)
            for k in CURVE_KEYS:
                reps[i][k].append(m[k])
    rows = []
    for p, m, r in zip(params, base, reps):
        row = {"param": p, "n_auto": m["n_auto"], "n_escalated": m["n_escalated"]}
        for k in CURVE_KEYS:
            row[k] = m[k]
            row[k + "_ci"] = _ci(r[k])
        rows.append(row)
    return rows


def spread(values: list[Optional[float]]) -> Optional[dict]:
    """min / mean / max over repeats (LLM nondeterminism); None entries are ignored."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}