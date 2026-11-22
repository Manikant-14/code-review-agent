"""Evaluation report (docs/evaluation.md): labels + metrics + intervals + operating curve.

Reads a reviewer JSONL and the judge's labels.jsonl, labels every finding (location match plus
judge verdict), and prints the metrics for a decision policy. Refuses to report while any
location-matched finding is still waiting for the judge, so a half-labeled run cannot produce
numbers. Dev and test are never pooled: --split picks one.

Policies: all_auto (no gate), score (auto-post when self-confidence >= --param), severity_scaled
(the gate's per-severity thresholds times --param), decisions (use each finding's own decision,
for gate output). The no-gate block is always printed first.

Usage (repo root, venv active):
  python -m benchmark.evaluate data/runs/pilot.jsonl
  python -m benchmark.evaluate data/runs/pilot.jsonl --policy score --param 0.85 --curve
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from benchmark import judge, labeling, metrics as mt

ROWS = [
    ("auto_post_precision", "auto-post precision"),
    ("escalation_rate", "escalation rate"),
    ("escalation_rate_pos", "  on positive runs"),
    ("escalation_rate_neg", "  on negative runs"),
    ("detection_recall", "detection recall"),
    ("autopost_recall", "auto-post recall"),
    ("fp_load_auto", "FP load, auto-posted / neg run"),
    ("fp_load_escalated", "FP load, escalated / neg run"),
    ("fix_flag_rate", "fix-flag rate (neg runs with a finding)"),
]
CURVE_PARAMS = {"score": [round(0.5 + 0.05 * i, 2) for i in range(11)],
                "severity_scaled": [round(0.5 + 0.1 * i, 1) for i in range(11)]}
KEY_FOR_SPREAD = ("auto_post_precision", "escalation_rate", "detection_recall", "autopost_recall", "fix_flag_rate")


def fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def fmt_entry(e: dict) -> str:
    ci = e.get("ci")
    ci_txt = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
    return f"{fmt(e['value'])}{ci_txt}"


def pending(findings) -> list:
    return [f for f in findings if f.is_tp is None]


def policy_fn(name: str):
    if name not in mt.POLICIES and name != "decisions":
        raise ValueError(f"unknown policy {name!r}")
    return None if name == "decisions" else mt.POLICIES[name]


def policy_block(title: str, runs, findings, auto_fn, param, B: int, seed: int) -> list[str]:
    fs = findings if auto_fn is None else mt.with_decisions(findings, auto_fn, param)
    s = mt.summarize(runs, fs, B, seed)
    c = s["counts"]
    out = [f"{title}", f"  findings {c['n_findings']}: auto-posted {c['n_auto']}, escalated {c['n_escalated']}"
           f"  |  runs: {c['n_pos_runs']} positive, {c['n_neg_runs']} negative"]
    for key, label in ROWS:
        e = s[key]
        n_txt = f"  (n={e['n_valid']} resamples)" if e["method"] == "bootstrap" and e["ci"] else ""
        out.append(f"  {label:42s} {fmt_entry(e):20s} {e['method']}{n_txt}")
    return out


def curve_block(kind: str, runs, findings, B: int, seed: int) -> list[str]:
    rows = mt.operating_curve(runs, findings, mt.POLICIES[kind], CURVE_PARAMS[kind], B, seed)
    name = "threshold" if kind == "score" else "scale"
    out = [f"OPERATING CURVE ({kind}; 95% bootstrap bands over cases)",
           f"  {name:9s} {'auto':>5s} {'esc':>4s}  {'auto-post precision':24s} {'escalation rate':20s} auto-post recall"]
    for r in rows:
        def cell(k):
            ci = r[k + "_ci"]
            return f"{fmt(r[k])}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
        out.append(f"  {r['param']:<9} {r['n_auto']:>5d} {r['n_escalated']:>4d}  {cell('auto_post_precision'):24s} "
                   f"{cell('escalation_rate'):20s} {cell('autopost_recall')}")
    return out


def spread_block(runs, findings, auto_fn, param) -> list[str]:
    samples = sorted({r.sample for r in runs})
    if len(samples) < 2:
        return []
    per = []
    for s in samples:
        rs = [r for r in runs if r.sample == s]
        fs = [f for f in findings if f.sample == s]
        per.append(mt.metrics(rs, fs if auto_fn is None else mt.with_decisions(fs, auto_fn, param)))
    out = [f"SPREAD ACROSS {len(samples)} REPEATS (min / mean / max of each repeat's point estimate)"]
    for k in KEY_FOR_SPREAD:
        sp = mt.spread([m[k] for m in per])
        out.append(f"  {k:24s} " + ("n/a" if sp is None else f"{sp['min']:.2f} / {sp['mean']:.2f} / {sp['max']:.2f}  (n={sp['n']})"))
    return out


def render_report(runs, findings, skipped, policy: str = "all_auto", param=None, curve: bool = False,
                  B: int = 2000, seed: int = 0, header: str = "") -> str:
    reasons = Counter(f.reason for f in findings)
    tp = sum(1 for f in findings if f.is_tp)
    lines = [header] if header else []
    lines.append(f"runs used {len(runs)}"
                 + (f"; skipped {len(skipped)}: " + "; ".join(f"{r} ({why})" for r, why in skipped[:5]) if skipped else ""))
    lines.append(f"findings {len(findings)}: true positive {tp}, false positive {len(findings) - tp}"
                 f"   [" + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + "]")
    lines.append("")
    lines += policy_block("NO GATE (everything auto-posted)", runs, findings, mt.all_auto, None, B, seed)
    if policy != "all_auto":
        lines.append("")
        label = {"score": f"SELF-CONFIDENCE BASELINE (auto-post when score >= {param})",
                 "severity_scaled": f"SEVERITY THRESHOLDS x {param}",
                 "decisions": "GATE DECISIONS (as recorded on each finding)"}[policy]
        lines += policy_block(label, runs, findings, policy_fn(policy), param, B, seed)
    if curve:
        lines.append("")
        lines += curve_block(policy if policy in CURVE_PARAMS else "score", runs, findings, B, seed)
    sp = spread_block(runs, findings, policy_fn(policy) if policy != "all_auto" else mt.all_auto, param)
    if sp:
        lines.append("")
        lines += sp
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Evaluation report for a reviewer JSONL.")
    ap.add_argument("runs", type=Path)
    ap.add_argument("--labels", type=Path, default=judge.LABELS_PATH)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--policy", default="all_auto", choices=["all_auto", "score", "severity_scaled", "decisions"])
    ap.add_argument("--param", type=float, default=None, help="threshold (score) or scale (severity_scaled)")
    ap.add_argument("--score-key", default="self_confidence", help="finding field the score policy compares")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.policy in ("score", "severity_scaled") and a.param is None:
        ap.error(f"--policy {a.policy} needs --param")
    records = labeling.load_latest_records(a.runs)
    cases = labeling.load_cases(split=a.split)
    verdicts = judge.load_verdicts(a.labels)
    runs, findings, skipped = labeling.build_dataset(records, cases, verdicts, a.score_key)
    if not runs:
        sys.exit(f"no usable runs for split {a.split!r} in {a.runs}")
    wait = pending(findings)
    if wait:
        sys.exit(f"{len(wait)} finding(s) are still waiting for the judge (e.g. {wait[0].id}). Run:\n"
                 f"  python -m benchmark.judge {a.runs}")
    print(render_report(runs, findings, skipped, a.policy, a.param, a.curve, a.boot, a.seed,
                        header=f"{a.runs}  split {a.split}  judge prompt {judge.JUDGE_PROMPT_VERSION}"))


if __name__ == "__main__":
    main(sys.argv[1:])