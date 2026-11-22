"""Labeling (docs/evaluation.md, "When a finding is correct").

Turns reviewer run records into labeled findings:
  negative run                      -> false positive ("negative_run")
  positive run, no location match   -> false positive ("location")
  positive run, location match      -> decided by the judge's verdict: yes = true positive,
                                       no = false positive, missing = pending (is_tp None)
The judge itself lives in benchmark/judge.py (step 6b); this module only consumes its verdicts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src import config

CASES_DIR = Path(__file__).resolve().parent / "cases"
DECISIONS = (None, "auto_post", "escalate")


def norm_path(p: str) -> str:
    return p.strip().replace("\\", "/").removeprefix("./")


def same_file(a: str, b: str) -> bool:
    """Equal, or one is a suffix of the other on a '/' boundary ("models.py" vs "httpie/models.py").

    Stricter than benchmark.bakeoff.is_hit, whose plain endswith lets "s.py" match "models.py".
    """
    a, b = norm_path(a), norm_path(b)
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def location_match(file: str, line: int, bug_locations: list[dict], window: Optional[int] = None) -> bool:
    """Same file as ANY bug location and line within [start - W, end + W] of it."""
    w = config.LOCATION_MATCH_WINDOW if window is None else window
    return any(same_file(file, loc["file"]) and loc["start"] - w <= line <= loc["end"] + w
               for loc in bug_locations)


@dataclass(frozen=True)
class Run:
    run_id: str
    case_id: str
    kind: str      # "positive" | "negative"
    sample: int


@dataclass(frozen=True)
class Labeled:
    id: str
    run_id: str
    case_id: str
    kind: str
    sample: int
    file: str
    line: int
    category: str
    severity: str
    score: Optional[float]       # the number a threshold policy compares (self_confidence by default)
    decision: Optional[str]      # "auto_post" | "escalate" | None until a gate or policy sets it
    location_match: bool
    is_tp: Optional[bool]        # None = waiting for the judge
    reason: str                  # negative_run | location | judge_yes | judge_no | pending_judge


def load_cases(cases_dir: Optional[Path] = None, split: Optional[str] = None) -> dict[str, dict]:
    """case_id -> {project, split, bug_locations}; optionally only one split."""
    out = {}
    for d in sorted((cases_dir or CASES_DIR).iterdir()):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if split and m["split"] != split:
            continue
        out[m["case_id"]] = {"project": m["project"], "split": m["split"],
                             "bug_locations": m["bug_locations"]}
    return out


def load_latest_records(path: Path) -> dict[str, dict]:
    """run_id -> the LAST record for it (a retry replaces an earlier error); bad lines are skipped."""
    latest = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec["run_id"]] = rec
    return latest


def judge_queue(records: dict[str, dict], cases: dict[str, dict],
                verdicts: Optional[dict[str, str]] = None, window: Optional[int] = None) -> list[dict]:
    """Positive-run findings that matched a bug location but have no judge verdict yet."""
    verdicts = verdicts or {}
    out = []
    for rid, rec in sorted(records.items()):
        if rec.get("status") != "ok" or rec["kind"] != "positive" or rec["case_id"] not in cases:
            continue
        locs = cases[rec["case_id"]]["bug_locations"]
        for f in rec["findings"]:
            if f["id"] not in verdicts and location_match(f["file"], f["line"], locs, window):
                out.append({"finding_id": f["id"], "case_id": rec["case_id"], "finding": f})
    return out


def build_dataset(records: dict[str, dict], cases: dict[str, dict],
                  verdicts: Optional[dict[str, str]] = None, score_key: str = "self_confidence",
                  window: Optional[int] = None):
    """-> (runs, findings, skipped). `cases` decides the split: runs of other cases are skipped.

    Runs in error are skipped and listed, never silently counted as zero findings.
    """
    verdicts = verdicts or {}
    bad = {v for v in verdicts.values() if v not in ("yes", "no")}
    if bad:
        raise ValueError(f"judge verdicts must be 'yes' or 'no', got {sorted(bad)}")
    runs: list[Run] = []
    findings: list[Labeled] = []
    skipped: list[tuple[str, str]] = []
    for rid, rec in sorted(records.items()):
        if rec["case_id"] not in cases:
            skipped.append((rid, "case not in this split"))
            continue
        if rec.get("status") != "ok":
            skipped.append((rid, f"run failed: {rec.get('error')}"))
            continue
        locs = cases[rec["case_id"]]["bug_locations"]
        runs.append(Run(rid, rec["case_id"], rec["kind"], rec["sample"]))
        for f in rec["findings"]:
            dec = f.get("decision")
            if dec not in DECISIONS:
                raise ValueError(f"finding {f['id']}: unknown decision {dec!r}")
            matched = location_match(f["file"], f["line"], locs, window)
            if rec["kind"] == "negative":
                is_tp, reason = False, "negative_run"
            elif not matched:
                is_tp, reason = False, "location"
            elif f["id"] in verdicts:
                is_tp = verdicts[f["id"]] == "yes"
                reason = "judge_yes" if is_tp else "judge_no"
            else:
                is_tp, reason = None, "pending_judge"
            findings.append(Labeled(
                id=f["id"], run_id=rid, case_id=rec["case_id"], kind=rec["kind"], sample=rec["sample"],
                file=f["file"], line=f["line"], category=f["category"], severity=f["severity"],
                score=f.get(score_key), decision=dec, location_match=matched, is_tp=is_tp, reason=reason))
    return runs, findings, skipped