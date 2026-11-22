"""Reviewer node (Phase 3 step 4): case -> prompt -> LLM -> findings -> JSONL.

One line per RUN in data/runs/reviewer.jsonl (a run = one diff reviewed once), including runs with
zero findings, because recall and false-positive load need them. Findings are state.Finding
objects with no signals or decision yet; later nodes fill those in. Runs that already succeeded are
skipped, so a re-run resumes. Use a different --out file for a different variant (e.g. no context).

Usage (repo root, venv active):
  python -m src.reviewer httpie-4                      # positive + negative run of one case
  python -m src.reviewer httpie-4 thefuck-29 --samples 2 --out data/runs/pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, field_validator

from src import config, llm
from src.review_input import build_review_input
from src.state import Finding

ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = ROOT / "data" / "runs" / "reviewer.jsonl"
PROMPT_VERSION = "v1"  # bump when SYSTEM changes; stored in every run record
MAX_FINDINGS = 3       # also enforced in code, not only in the prompt

SYSTEM = (
    "You are a careful senior code reviewer. You are shown a code change as a diff. Context and "
    "added lines carry their line number in the NEW version of the file; removed lines are marked "
    "with '-' and have no number. After the diff you may get 'context' blocks: the surrounding "
    "code of the new version, with line numbers. Use them only to understand the change; report "
    "defects in the changed lines.\n"
    "Report only real defects that this change introduces and that would cause wrong behaviour or "
    "an error at runtime. Do not comment on style, naming, tests, documentation or possible "
    "improvements. If you find no real defect, return an empty list. Report at most 3 findings, "
    "most serious first. For a line that was removed, give the number of the nearest numbered "
    "line next to it.\n"
    "For each finding give: category, one of logic, security, error_handling, concurrency, "
    "performance, style; severity, one of low, medium, high (how costly the defect is if it "
    "ships); message (what is wrong and why); expected_failure (the exception type or wrong "
    "behaviour a test would show, for example \"KeyError\" or \"returns an empty list\"), or null "
    "if no runtime failure can be shown; self_confidence, your honest probability from 0 to 1 that "
    "this finding is a real defect.\n"
    'Reply with only a JSON object like {"findings": [{"file": "path/to/file.py", "line": 12, '
    '"category": "logic", "severity": "medium", "message": "what is wrong and why", '
    '"expected_failure": "IndexError", "self_confidence": 0.7}]}'
)


class ReviewerFinding(BaseModel):
    file: str
    line: int
    category: config.Category
    severity: config.Severity
    message: str
    expected_failure: Optional[str] = None
    self_confidence: float

    @field_validator("category", "severity", mode="before")
    @classmethod
    def _norm_enum(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v

    @field_validator("expected_failure", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a"):
            return None
        return v

    @field_validator("self_confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return v
        if 1 < x <= 100:  # the model answered in percent
            x /= 100
        return min(1.0, max(0.0, x))


class Review(BaseModel):
    findings: list[ReviewerFinding]


def make_run_id(case_id: str, kind: str, sample: int) -> str:
    return f"{case_id}:{kind}:s{sample}"


def _norm_path(p: str) -> str:
    return p.strip().replace("\\", "/").removeprefix("./")


def to_findings(run_id: str, items: list[ReviewerFinding]) -> list[Finding]:
    return [
        Finding(id=f"{run_id}:f{i}", file=_norm_path(f.file), line=f.line, category=f.category,
                severity=f.severity, message=f.message.strip(), expected_failure=f.expected_failure,
                self_confidence=f.self_confidence)
        for i, f in enumerate(items[:MAX_FINDINGS])
    ]


def review_run(case_id: str, kind: str, sample: int = 0, with_context: bool = True,
               cases_dir: Optional[Path] = None, read_file: Optional[Callable] = None) -> dict:
    """Review one diff once; returns the run record (not yet written anywhere)."""
    ri = build_review_input(case_id, kind, with_context=with_context, cases_dir=cases_dir,
                            read_file=read_file)
    run_id = make_run_id(case_id, kind, sample)
    review = llm.call_llm("reviewer", ri.text, system=SYSTEM, schema=Review, sample=sample)
    return {
        "run_id": run_id, "case_id": case_id, "kind": kind, "sample": sample, "status": "ok",
        "error": None, "model": config.MODELS["reviewer"], "temperature": config.TEMPERATURES["reviewer"],
        "prompt_version": PROMPT_VERSION, "with_context": ri.with_context,
        "context_lines": ri.context_lines, "est_tokens": ri.est_tokens, "notes": ri.notes,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "findings": [f.model_dump(mode="json") for f in to_findings(run_id, review.findings)],
    }


def _done_run_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line: that run is simply redone
            if rec.get("status") == "ok":
                done.add(rec["run_id"])
    return done


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def run_reviewer(case_ids: list[str], kinds=("positive", "negative"), samples: int = 1,
                 with_context: bool = True, out_path: Path = RUNS_PATH,
                 review_fn: Callable = review_run, on_record: Optional[Callable] = None) -> dict:
    """Run every (case, kind, sample) not already done; append one JSONL line each."""
    done = _done_run_ids(out_path)
    summary = {"ok": 0, "error": 0, "skipped": 0, "findings": 0, "stopped": None}
    for cid in case_ids:
        for kind in kinds:
            for s in range(samples):
                rid = make_run_id(cid, kind, s)
                if rid in done:
                    summary["skipped"] += 1
                    continue
                try:
                    rec = review_fn(cid, kind, s, with_context)
                except llm.DailyQuotaExhausted as e:
                    summary["stopped"] = f"daily quota exhausted at {rid}: {str(e)[:100]}"
                    return summary
                except Exception as e:  # noqa: BLE001  (record it, keep going)
                    rec = {"run_id": rid, "case_id": cid, "kind": kind, "sample": s, "status": "error",
                           "error": f"{type(e).__name__}: {str(e)[:200]}", "prompt_version": PROMPT_VERSION,
                           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "findings": []}
                _append(out_path, rec)
                summary[rec["status"]] += 1
                summary["findings"] += len(rec["findings"])
                if on_record:
                    on_record(rec)
    return summary


def _print_record(rec: dict) -> None:
    if rec["status"] != "ok":
        print(f"{rec['run_id']:28s} ERROR {rec['error']}")
        return
    print(f"{rec['run_id']:28s} ok  findings={len(rec['findings'])}  ~{rec['est_tokens']} tok"
          f"{'  ctx' if rec['with_context'] else '  no-ctx'}")
    for f in rec["findings"]:
        print(f"    L{f['line']} {f['file']} [{f['category']}/{f['severity']}] "
              f"conf={f['self_confidence']:.2f} exp={f['expected_failure']}: {f['message'][:110]}")


def main(argv: list[str]) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="+", help="case ids, e.g. httpie-4")
    ap.add_argument("--kinds", nargs="+", default=["positive", "negative"], choices=["positive", "negative"])
    ap.add_argument("--samples", type=int, default=1, help="repeat each run this many times")
    ap.add_argument("--no-context", action="store_true", help="diff only (an ablation: use another --out)")
    ap.add_argument("--out", type=Path, default=RUNS_PATH)
    a = ap.parse_args(argv)
    model = config.MODELS["reviewer"]
    # Groq's 8K tokens/min is the real limit and reasoning tokens count: 2 RPM with context.
    config.RPM_LIMITS[model] = min(config.RPM_LIMITS.get(model, 2), 2)
    print(f"reviewer {model}  prompt {PROMPT_VERSION}  out {a.out}\n")
    s = run_reviewer(a.cases, tuple(a.kinds), a.samples, not a.no_context, a.out, on_record=_print_record)
    print(f"\nok={s['ok']} error={s['error']} skipped={s['skipped']} findings={s['findings']}")
    if s["stopped"]:
        print("STOPPED: " + s["stopped"] + "\nRe-run the same command tomorrow: finished runs are skipped.")


if __name__ == "__main__":
    main(sys.argv[1:])