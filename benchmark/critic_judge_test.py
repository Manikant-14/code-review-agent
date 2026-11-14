"""Step 2b: which model should be the critic and which the judge?

Reuses the real findings gpt-oss-120b produced in the bake-off (cached, free): 4 true findings and
3 false ones. Each candidate then plays both roles on those same findings:
  critic: sees the diff + one finding, should rebut the false ones and let the true ones stand
  judge : sees the official fix + one finding, should call true ones correct and false ones incorrect
Usage (repo root, venv active):
  python -m benchmark.critic_judge_test groq:qwen/qwen3.8-27b
  python -m benchmark.critic_judge_test gemini:gemma-4-31b-it
Every call is cached, so re-running costs nothing.
"""
import json
import sys
import time
from typing import Literal

from pydantic import BaseModel, field_validator

from benchmark import bakeoff
from benchmark.bakeoff import CASES_DIR, annotate_diff, is_hit
from src import config, llm

REVIEWER = "groq:openai/gpt-oss-120b"
# Checked by hand: the 120b finding on thefuck-10 points at the removed block (line 14), which is
# the real defect, but the recorded bug location only covers the added lines (21-24).
TRUE_OVERRIDES = {("thefuck-10", "positive")}

CRITIC_SYSTEM = (
    "You are a skeptical senior engineer checking another reviewer's claim. You see a diff (context "
    "and added lines carry their line number in the NEW version of the file; removed lines are marked "
    "'-') and one claimed defect. Look for the strongest concrete reason the claim is wrong: the "
    "reviewer misread the diff, assumed something the code does not do, or the behaviour does not "
    "actually change or break. Do not rebut just because you can imagine a counterargument: rebut "
    "only if you have a concrete one. If you cannot find a concrete rebuttal, the claim stands.\n"
    'Reply with only a JSON object like {"rebuttal": "one or two sentences", "verdict": "agree"} '
    'where verdict is "rebut" if you found a concrete rebuttal, otherwise "agree".'
)
JUDGE_SYSTEM = (
    "You grade whether a reported finding describes a real defect, given the official fix. You see "
    "the official patch that fixed the bug (a diff from the buggy version to the fixed version) and "
    "one finding from a reviewer. The finding is correct only if it describes the defect that the "
    "patch fixes: the same faulty behaviour in the buggy code. A finding that criticises the fixed "
    "code, treats the patch's own change as a problem, or is about something unrelated is incorrect. "
    "Ignore line numbers; judge by behaviour.\n"
    'Reply with only a JSON object like {"reason": "one sentence", "verdict": "correct"} where '
    'verdict is "correct" or "incorrect".'
)


def _as_text(v):
    """Some models return null or a list for a free-text field: keep the verdict, not the format."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v if isinstance(v, str) else str(v)


class Critique(BaseModel):
    rebuttal: str = ""
    verdict: Literal["agree", "rebut"]

    @field_validator("rebuttal", mode="before")
    @classmethod
    def _text(cls, v):
        return _as_text(v)


class Grade(BaseModel):
    reason: str = ""
    verdict: Literal["correct", "incorrect"]

    @field_validator("reason", mode="before")
    @classmethod
    def _text(cls, v):
        return _as_text(v)


def reviewer_findings(case_ids: list[str]) -> list[dict]:
    """The 120b findings for the bake-off cases (served from the cache), with a true/false label."""
    config.MODELS["reviewer"] = REVIEWER
    config.RPM_LIMITS.setdefault(REVIEWER, 3)
    out = []
    for cid in case_ids:
        meta = json.loads((CASES_DIR / cid / "meta.json").read_text(encoding="utf-8"))
        for kind in ("positive", "negative"):
            diff = (CASES_DIR / cid / f"{kind}.diff").read_text(encoding="utf-8")
            review = llm.call_llm("reviewer", "Review this change.\n\n" + annotate_diff(diff),
                                  system=bakeoff.SYSTEM, schema=bakeoff.Review)
            for f in review.findings:
                true = kind == "positive" and (
                    is_hit(f, meta["bug_locations"], config.LOCATION_MATCH_WINDOW)
                    or (cid, kind) in TRUE_OVERRIDES)
                out.append({"case": cid, "kind": kind, "finding": f, "true": true})
    return out


def run(model: str, items: list[dict]) -> dict:
    config.MODELS["critic"] = config.MODELS["judge"] = model
    config.RPM_LIMITS.setdefault(model, bakeoff.RPM_BY_PROVIDER.get(model.partition(":")[0], 4))
    offset = llm.LOG_PATH.stat().st_size if llm.LOG_PATH.exists() else 0
    res = {"model": model, "critic_right": 0, "judge_right": 0, "errors": [], "rows": []}
    for it in items:
        cid, kind, f = it["case"], it["kind"], it["finding"]
        shown = annotate_diff((CASES_DIR / cid / f"{kind}.diff").read_text(encoding="utf-8"))
        fix = annotate_diff((CASES_DIR / cid / "negative.diff").read_text(encoding="utf-8"))
        claim = f"({f.file}, line {f.line}):\n{f.message}"
        row = {"case": cid, "kind": kind, "true": it["true"], "critic": "ERR", "judge": "ERR"}
        for role, system, prompt, schema, good in (
            ("critic", CRITIC_SYSTEM, f"Diff:\n{shown}\n\nClaimed defect {claim}", Critique,
             "agree" if it["true"] else "rebut"),
            ("judge", JUDGE_SYSTEM, f"Official fix (buggy -> fixed):\n{fix}\n\nFinding {claim}", Grade,
             "correct" if it["true"] else "incorrect"),
        ):
            try:
                v = llm.call_llm(role, prompt, system=system, schema=schema).verdict
            except llm.DailyQuotaExhausted as e:
                res["errors"].append(f"DAILY QUOTA ({str(e)[:80]})")
                return _finish(res, offset, model)
            except Exception as e:  # noqa: BLE001
                res["errors"].append(f"{cid}/{kind}/{role}: {type(e).__name__}: {str(e)[:100]}")
                continue
            row[role] = v + (" ok" if v == good else " WRONG")
            res[f"{role}_right"] += v == good
        res["rows"].append(row)
    return _finish(res, offset, model)


def _finish(res: dict, offset: int, model: str) -> dict:
    rows = bakeoff._log_rows_since(offset, model)
    n = max(len(rows), 1)
    res["avg_latency"] = round(sum(r.get("latency_s") or 0 for r in rows) / n, 1)
    res["total_tokens"] = sum((r.get("prompt_tokens") or 0) + (r.get("output_tokens") or 0) for r in rows)
    return res


def main(argv: list[str]) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not argv:
        sys.exit("usage: python -m benchmark.critic_judge_test provider:model [provider:model ...]")
    items = reviewer_findings(bakeoff.pick_cases(5))
    n_true = sum(i["true"] for i in items)
    print(f"{len(items)} reviewer findings: {n_true} true, {len(items) - n_true} false\n")
    results = []
    for m in argv:
        print(f"== {m}")
        t0 = time.time()
        r = run(m, items)
        print(f"  {'case':11s} {'diff':9s} {'finding':8s} {'critic':12s} judge")
        for row in r["rows"]:
            print(f"  {row['case']:11s} {row['kind']:9s} {'TRUE' if row['true'] else 'false':8s} "
                  f"{row['critic']:12s} {row['judge']}")
        for e in r["errors"]:
            print("  ! " + e)
        print(f"  ({time.time() - t0:.0f}s)\n")
        results.append(r)
    print("SUMMARY (right = true findings kept / false findings rejected)")
    print(f"{'model':36s} {'critic right':>13s} {'judge right':>12s} {'errors':>7s} {'secs/call':>10s}")
    for r in results:
        print(f"{r['model']:36s} {str(r['critic_right']) + '/' + str(len(items)):>13s} "
              f"{str(r['judge_right']) + '/' + str(len(items)):>12s} {len(r['errors']):>7d} {r['avg_latency']:>10.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
