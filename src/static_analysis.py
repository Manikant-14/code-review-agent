"""Static analysis node (Phase 4 step 1): ruff on the full new-version file, filtered to changed lines.

Signal (docs/signals.md): static_overlap = 1.0 if ruff flags a line within +/-STATIC_OVERLAP_LINE_WINDOW
of the finding, otherwise unavailable (None, never 0.0). Rule set and filtering are fixed in
docs/evaluation.md ("Static analysis").

Pieces, all pure except run_ruff:
  added_lines(diff)            file -> new-side line numbers the diff added
  run_ruff(source, path)       ruff over stdin, slim issue dicts (raises StaticError if ruff cannot run)
  filter_changed(issues, ...)  keep issues whose row span touches an added line
  apply_static(findings, ...)  set Finding.static_overlap
  static_node(state)           LangGraph node over ReviewState

Usage (repo root, venv active): replay a reviewer runs file and persist the signal for offline gate replay
  python -m src.static_analysis data\\runs\\pilot.jsonl --out data\\signals\\static.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from unidiff import PatchSet

from src import config
from src.review_input import CASES_DIR, git_show
from src.state import Finding, ReviewState

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_PATH = ROOT / "data" / "signals" / "static.jsonl"


class StaticError(Exception):
    """ruff could not run or returned something unusable: the signal is unavailable, not zero."""


def _ruff(*args: str, source: Optional[str] = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "ruff", *args]
    try:
        return subprocess.run(cmd, input=None if source is None else source.encode("utf-8"),
                              capture_output=True, timeout=config.STATIC_RUFF_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise StaticError(f"ruff could not run: {type(e).__name__}: {str(e)[:150]}") from e


def ruff_version() -> Optional[str]:
    try:
        r = _ruff("--version")
    except StaticError:
        return None
    return r.stdout.decode("utf-8", errors="replace").strip() if r.returncode == 0 else None


def _slim(d: dict) -> dict:
    row = d["location"]["row"]
    end = (d.get("end_location") or {}).get("row", row)
    return {"code": d.get("code"), "name": d.get("name"), "message": d.get("message"),
            "row": row, "end_row": max(row, end)}


def run_ruff(source: str, path: str, rules: tuple[str, ...] = config.STATIC_RULES) -> list[dict]:
    """Lint one file's full contents. --isolated: the project's own ruff config must not change results."""
    r = _ruff("check", "--isolated", "--select", ",".join(rules), "--output-format", "json",
              "--stdin-filename", path, "-", source=source)
    if r.returncode not in (0, 1):  # 0 clean, 1 issues found, anything else is a failure
        raise StaticError(f"ruff exit {r.returncode}: {r.stderr.decode('utf-8', errors='replace')[:200]}")
    try:
        return [_slim(d) for d in json.loads(r.stdout.decode("utf-8"))]
    except (ValueError, KeyError, TypeError) as e:
        raise StaticError(f"unreadable ruff output: {type(e).__name__}: {str(e)[:100]}") from e


def added_lines(diff_text: str) -> dict[str, set[int]]:
    """New-side line numbers added by the diff, per Python file. Deleted files and non-.py files are skipped."""
    out: dict[str, set[int]] = {}
    for pf in PatchSet(diff_text):
        if pf.is_removed_file or not pf.path.endswith(".py"):
            continue
        lines = {ln.target_line_no for h in pf for ln in h if ln.is_added}
        if lines:
            out[pf.path] = lines
    return out


def filter_changed(issues: list[dict], changed: set[int]) -> list[dict]:
    return [i for i in issues if any(i["row"] <= n <= i["end_row"] for n in changed)]


def overlaps(issues: list[dict], line: int, window: int) -> list[dict]:
    lo, hi = line - window, line + window
    return [i for i in issues if i["row"] <= hi and i["end_row"] >= lo]


def apply_static(findings: list[Finding], issues_by_file: dict[str, list[dict]],
                 window: int = config.STATIC_OVERLAP_LINE_WINDOW) -> list[Finding]:
    out = []
    for f in findings:
        hit = overlaps(issues_by_file.get(f.file, []), f.line, window)
        out.append(f.model_copy(update={"static_overlap": 1.0 if hit else None}))
    return out


def static_node(state: ReviewState) -> dict:
    """LangGraph node. A file that cannot be read or linted leaves its findings unavailable and logs an error."""
    errors = list(state.get("errors", []))
    issues: list[dict] = []
    by_file: dict[str, list[dict]] = {}
    for path, changed in added_lines(state["diff"]).items():
        source = state["files"].get(path)
        if source is None:
            errors.append(f"static: no source for {path}")
            continue
        try:
            found = filter_changed(run_ruff(source, path), changed)
        except StaticError as e:
            errors.append(f"static: {path}: {e}")
            continue
        by_file[path] = found
        issues += [{**i, "file": path} for i in found]
    return {"static_issues": issues, "findings": apply_static(state["findings"], by_file), "errors": errors}


# ---------------------------------------------------------------- benchmark replay

def state_for_run(rec: dict, cases_dir: Optional[Path] = None,
                  show: Callable[[str, str, str], Optional[str]] = git_show) -> ReviewState:
    """ReviewState for one reviewer run record: the diff it saw and the new-version files at its commit."""
    case_dir = (cases_dir or CASES_DIR) / rec["case_id"]
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    diff = (case_dir / f"{rec['kind']}.diff").read_text(encoding="utf-8")
    commit = meta["buggy_commit"] if rec["kind"] == "positive" else meta["fixed_commit"]
    files = {}
    for path in added_lines(diff):
        src = show(meta["project"], commit, path)
        if src is not None:
            files[path] = src
    return {"pr_ref": rec["run_id"], "diff": diff, "files": files, "static_issues": [],
            "findings": [Finding(**f) for f in rec.get("findings", [])], "errors": []}


def signal_record(rec: dict, out: dict, version: Optional[str]) -> dict:
    matches = {f.id: overlaps([i for i in out["static_issues"] if i["file"] == f.file], f.line,
                              config.STATIC_OVERLAP_LINE_WINDOW) for f in out["findings"]}
    return {"run_id": rec["run_id"], "ruff_version": version, "rules": list(config.STATIC_RULES),
            "window": config.STATIC_OVERLAP_LINE_WINDOW, "issues": out["static_issues"], "errors": out["errors"],
            "findings": [{"id": f.id, "file": f.file, "line": f.line, "static_overlap": f.static_overlap,
                          "matches": [{"code": m["code"], "row": m["row"]} for m in matches[f.id]]}
                         for f in out["findings"]]}


def replay(runs_path: Path, out_path: Path, cases_dir: Optional[Path] = None,
           show: Callable[[str, str, str], Optional[str]] = git_show) -> dict:
    """Run the static node over every ok run in a reviewer JSONL; rewrites out_path (deterministic, fast)."""
    version = ruff_version()
    if version is None:
        raise StaticError("ruff does not run here (python -m ruff --version failed)")
    recs = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "ok":
            recs.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"runs": 0, "findings": 0, "overlap": 0, "issues": 0, "errors": 0, "codes": Counter(), "hits": []}
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in recs:
            out = static_node(state_for_run(rec, cases_dir, show))
            sr = signal_record(rec, out, version)
            fh.write(json.dumps(sr, ensure_ascii=False) + "\n")
            stats["runs"] += 1
            stats["findings"] += len(sr["findings"])
            stats["issues"] += len(sr["issues"])
            stats["errors"] += len(sr["errors"])
            stats["codes"].update(i["code"] for i in sr["issues"])
            for f in sr["findings"]:
                if f["static_overlap"] == 1.0:
                    stats["overlap"] += 1
                    stats["hits"].append((f["id"], f["line"], f["matches"]))
    stats["ruff_version"] = version
    return stats


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, help="reviewer JSONL, e.g. data/runs/pilot.jsonl")
    ap.add_argument("--out", type=Path, default=SIGNALS_PATH)
    a = ap.parse_args(argv)
    s = replay(a.runs, a.out)
    print(f"{s['ruff_version']}  rules={','.join(config.STATIC_RULES)}  window=+/-{config.STATIC_OVERLAP_LINE_WINDOW}")
    print(f"runs={s['runs']} findings={s['findings']} with static_overlap={s['overlap']} "
          f"ruff issues on changed lines={s['issues']} errors={s['errors']}")
    if s["codes"]:
        print("issue codes: " + ", ".join(f"{c}={n}" for c, n in s["codes"].most_common()))
    for fid, line, m in s["hits"]:
        print(f"  overlap  {fid} L{line}  <- " + ", ".join(f"{x['code']}@{x['row']}" for x in m))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
