"""LLM judge (docs/evaluation.md, "Judge rules"): does a location-matched finding describe the
defect that the patch fixes?

The judge model (config.MODELS["judge"], different from reviewer and critic) sees ONLY the
buggy->fixed patch and the finding's file, line, message and expected failure. It never sees
confidence, severity, category, signals or the gate decision. Verdicts are appended to
data/runs/labels.jsonl, one line per finding, and a re-run skips findings already judged with the
current prompt version. Only findings from benchmark.labeling.judge_queue are sent.

PROMPT_VERSION is part of every record: change the prompt only before any judged results are
seen, otherwise log a deviation (old verdicts are ignored once the version changes).

Usage (repo root, venv active):
  python -m benchmark.judge data/runs/pilot.jsonl --dry-run
  python -m benchmark.judge data/runs/pilot.jsonl [--workers 4] [--labels data/runs/labels.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, field_validator

from benchmark import labeling
from src import config, llm

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "runs" / "labels.jsonl"
JUDGE_PROMPT_VERSION = "j1"

SYSTEM = (
    "You are a strict evaluator of automated code review findings. You are given (1) a patch that "
    "fixes a bug, shown as a unified diff from the buggy version (lines starting with '-') to the "
    "fixed version (lines starting with '+'), and (2) one finding written by an automated reviewer "
    "about the buggy version. Decide whether the finding describes the defect that this patch "
    "fixes: the same underlying problem, in the same place, with a consistent explanation of why "
    "it is wrong. Answer \"no\" if the finding is about a different problem, is vague or generic, "
    "only says that the code changed, or complains about something the patch does not address. A "
    "finding that identifies the right defect but guesses a slightly different symptom still "
    "counts as \"yes\". Reply with only a JSON object like "
    '{"verdict": "yes", "reason": "one sentence"}'
)


class Verdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: str

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


def build_prompt(patch_text: str, finding: dict) -> str:
    """Patch plus finding text only. Confidence, severity, category and decision stay out."""
    return (
        f"Patch (buggy -> fixed):\n{patch_text.rstrip()}\n\n"
        f"Finding:\nfile: {finding['file']}\nline: {finding['line']}\n"
        f"message: {finding['message']}\n"
        f"expected failure: {finding.get('expected_failure') or 'none given'}\n\n"
        "Does this finding describe the defect fixed by this patch?"
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def judge_finding(item: dict, cases_dir: Optional[Path] = None) -> dict:
    """Judge one queue item; returns the label record (not yet written anywhere)."""
    patch = ((cases_dir or labeling.CASES_DIR) / item["case_id"] / "negative.diff").read_text(encoding="utf-8")
    v = llm.call_llm("judge", build_prompt(patch, item["finding"]), system=SYSTEM, schema=Verdict)
    return {"finding_id": item["finding_id"], "case_id": item["case_id"], "verdict": v.verdict,
            "reason": v.reason.strip(), "judge_model": config.MODELS["judge"],
            "prompt_version": JUDGE_PROMPT_VERSION, "status": "ok", "error": None, "created_utc": _now()}


def load_verdicts(path: Path, version: str = JUDGE_PROMPT_VERSION) -> dict[str, str]:
    """finding_id -> "yes"/"no" from successful records of this prompt version (the last one wins)."""
    out: dict[str, str] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok" and rec.get("prompt_version") == version:
                out[rec["finding_id"]] = rec["verdict"]
    return out


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def run_judge(queue: list[dict], out_path: Path = LABELS_PATH, workers: int = 4,
              judge_fn: Callable = judge_finding, on_record: Optional[Callable] = None) -> dict:
    """Judge every queue item without a current-version verdict, `workers` at a time.

    Only the calling thread writes the file. A daily-quota error stops new work; other errors are
    recorded and retried on the next run.
    """
    done = load_verdicts(out_path)
    todo = [q for q in queue if q["finding_id"] not in done]
    summary = {"ok": 0, "error": 0, "skipped": len(queue) - len(todo), "stopped": None}
    if not todo:
        return summary
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(judge_fn, q): q for q in todo}
        for fut in as_completed(futures):
            if fut.cancelled():
                continue
            q = futures[fut]
            try:
                rec = fut.result()
            except llm.DailyQuotaExhausted as e:
                if summary["stopped"] is None:
                    summary["stopped"] = f"daily quota exhausted: {str(e)[:100]}"
                    for other in futures:
                        other.cancel()
                continue
            except Exception as e:  # noqa: BLE001  (record it, keep going)
                rec = {"finding_id": q["finding_id"], "case_id": q["case_id"], "verdict": None, "reason": None,
                       "judge_model": config.MODELS["judge"], "prompt_version": JUDGE_PROMPT_VERSION,
                       "status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}", "created_utc": _now()}
            _append(out_path, rec)
            summary[rec["status"]] += 1
            if on_record:
                on_record(rec)
    return summary


def _cached(item: dict) -> bool:
    patch = (labeling.CASES_DIR / item["case_id"] / "negative.diff").read_text(encoding="utf-8")
    key = llm._key(config.MODELS["judge"], config.TEMPERATURES["judge"], SYSTEM,
                   build_prompt(patch, item["finding"]), Verdict, 0)
    return (llm.CACHE_DIR / key[:2] / f"{key}.json").exists()


def _print_record(rec: dict) -> None:
    if rec["status"] != "ok":
        print(f"{rec['finding_id']:34s} ERROR {rec['error']}")
    else:
        print(f"{rec['finding_id']:34s} {rec['verdict']:3s}  {rec['reason'][:110]}")


def main(argv: list[str]) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="Judge location-matched findings of a reviewer JSONL.")
    ap.add_argument("runs", type=Path, help="reviewer JSONL, e.g. data/runs/pilot.jsonl")
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dry-run", action="store_true", help="show what would be judged; no API calls")
    a = ap.parse_args(argv)
    records = labeling.load_latest_records(a.runs)
    cases = labeling.load_cases(split=a.split)
    queue = labeling.judge_queue(records, cases, load_verdicts(a.labels))
    print(f"judge {config.MODELS['judge']}  prompt {JUDGE_PROMPT_VERSION}  split {a.split}  labels {a.labels}")
    print(f"{len(queue)} finding(s) to judge\n")
    if a.dry_run:
        for q in queue:
            f = q["finding"]
            print(f"{q['finding_id']:34s} {'cached' if _cached(q) else 'NEW   '}  L{f['line']} {f['message'][:80]}")
        print(f"\ndry run: {sum(not _cached(q) for q in queue)} call(s) would be NEW")
        return
    s = run_judge(queue, a.labels, a.workers, on_record=_print_record)
    left = labeling.judge_queue(records, cases, load_verdicts(a.labels))
    print(f"\nok={s['ok']} error={s['error']} skipped={s['skipped']} still unjudged={len(left)}")
    if s["stopped"]:
        print("STOPPED: " + s["stopped"] + "\nRe-run the same command later: judged findings are skipped.")


if __name__ == "__main__":
    main(sys.argv[1:])