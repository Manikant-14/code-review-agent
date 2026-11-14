"""Step 2 bake-off: pick the reviewer/critic/judge models with real data, not guesses.

Runs N dev cases (spread across projects, fixed seed) through each candidate model as a plain
reviewer, on the positive diff (real bug: did it find it?) and the negative diff (the fix: it
should stay quiet). Location-only scoring: a cheap proxy, the real judge comes later.

Usage (repo root, venv active):
  python -m benchmark.bakeoff groq:openai/gpt-oss-120b groq:openai/gpt-oss-20b
  python -m benchmark.bakeoff gemini:gemma-4-31b-it --n 5
Every call goes through the disk cache, so re-running costs nothing.
"""
import json
import random
import sys
import time
from pathlib import Path

from pydantic import BaseModel
from unidiff import PatchSet

from src import config, llm

CASES_DIR = Path(__file__).resolve().parent / "cases"
SEED = 7
# Requests per minute used while benchmarking. Groq's 8K tokens/min is the real limit there.
RPM_BY_PROVIDER = {"groq": 2, "gemini": 6}

SYSTEM = (
    "You are a careful senior code reviewer. You are shown a code change as a diff. Context and "
    "added lines carry their line number in the NEW version of the file; removed lines are marked "
    "with '-' and have no number.\n"
    "Report only real defects that this change introduces and that would cause wrong behaviour or "
    "an error at runtime. Do not comment on style, naming, tests, documentation or possible "
    "improvements. If you find no real defect, return an empty list. Report at most 3 findings, "
    "most serious first. For a line that was removed, give the number of the nearest numbered "
    "line next to it.\n"
    'Reply with only a JSON object like {"findings": [{"file": "path/to/file.py", "line": 12, '
    '"message": "what is wrong and why"}]}'
)


class Finding(BaseModel):
    file: str
    line: int
    message: str


class Review(BaseModel):
    findings: list[Finding]


def annotate_diff(diff_text: str) -> str:
    """Render a diff with new-file line numbers; index lines are dropped."""
    out = []
    for pf in PatchSet(diff_text):
        out.append(f"=== {pf.path}")
        for hunk in pf:
            out.append(f"@@ new lines {hunk.target_start}-{hunk.target_start + hunk.target_length - 1} @@")
            for line in hunk:
                text = line.value.rstrip("\n")
                if line.is_removed:
                    out.append(f"    -| {text}")
                elif line.is_added:
                    out.append(f"{line.target_line_no:5d}+| {text}")
                else:
                    out.append(f"{line.target_line_no:5d} | {text}")
    return "\n".join(out)


def pick_cases(n: int) -> list[str]:
    metas = {}
    for d in sorted(CASES_DIR.iterdir()):
        mp = d / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            metas.setdefault(m["project"], []).append(d.name)
    rng = random.Random(SEED)
    for ids in metas.values():
        rng.shuffle(ids)
    order = []
    while len(order) < n and any(metas.values()):
        for proj in sorted(metas):
            if metas[proj] and len(order) < n:
                order.append(metas[proj].pop())
    return order


def is_hit(f: Finding, locations: list[dict], w: int) -> bool:
    path = f.file.replace("\\", "/").removeprefix("./")
    for loc in locations:
        same_file = loc["file"].endswith(path) or path.endswith(loc["file"])
        if same_file and loc["start"] - w <= f.line <= loc["end"] + w:
            return True
    return False


def _log_rows_since(offset: int, model: str) -> list[dict]:
    if not llm.LOG_PATH.exists():
        return []
    with llm.LOG_PATH.open("rb") as fh:
        fh.seek(offset)
        rows = [json.loads(x) for x in fh.read().decode("utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("model") == model and not r.get("cached")]


def _msgs(fs) -> str:
    """One indented line per finding: where it points and what it says."""
    return "".join(f"\n      L{f.line} {f.file}: {f.message[:150]}" for f in fs)


def run_model(model: str, case_ids: list[str]) -> dict:
    config.MODELS["reviewer"] = model
    config.RPM_LIMITS.setdefault(model, RPM_BY_PROVIDER.get(model.partition(":")[0], 4))
    offset = llm.LOG_PATH.stat().st_size if llm.LOG_PATH.exists() else 0
    s = {"model": model, "pos_ok": 0, "pos_hit": 0, "pos_extra": 0, "neg_ok": 0, "neg_fp_cases": 0,
         "neg_findings": 0, "errors": [], "detail": []}
    for cid in case_ids:
        meta = json.loads((CASES_DIR / cid / "meta.json").read_text(encoding="utf-8"))
        for kind in ("positive", "negative"):
            diff = (CASES_DIR / cid / f"{kind}.diff").read_text(encoding="utf-8")
            prompt = "Review this change.\n\n" + annotate_diff(diff)
            try:
                review = llm.call_llm("reviewer", prompt, system=SYSTEM, schema=Review)
            except llm.DailyQuotaExhausted as e:
                s["errors"].append(f"{cid}/{kind}: DAILY QUOTA ({str(e)[:80]})")
                print(f"  {model}: daily quota exhausted, stopping this model")
                return _finish(s, offset, model)
            except Exception as e:  # noqa: BLE001  (record any failure, keep going)
                s["errors"].append(f"{cid}/{kind}: {type(e).__name__}: {str(e)[:100]}")
                s["detail"].append(f"{cid:12s} {kind:8s} ERROR")
                continue
            fs = review.findings
            if kind == "positive":
                hits = [f for f in fs if is_hit(f, meta["bug_locations"], config.LOCATION_MATCH_WINDOW)]
                s["pos_ok"] += 1
                s["pos_hit"] += bool(hits)
                s["pos_extra"] += len(fs) - len(hits)
                s["detail"].append(f"{cid:12s} positive {'HIT ' if hits else 'miss'} findings={len(fs)}" + _msgs(fs))
            else:
                s["neg_ok"] += 1
                s["neg_fp_cases"] += bool(fs)
                s["neg_findings"] += len(fs)
                s["detail"].append(f"{cid:12s} negative {'quiet' if not fs else 'FALSE POSITIVE x' + str(len(fs))}" + _msgs(fs))
    return _finish(s, offset, model)


def _finish(s: dict, offset: int, model: str) -> dict:
    rows = _log_rows_since(offset, model)
    n = max(len(rows), 1)
    s["calls_billed"] = len(rows)
    s["avg_prompt"] = round(sum(r.get("prompt_tokens") or 0 for r in rows) / n)
    s["avg_output"] = round(sum(r.get("output_tokens") or 0 for r in rows) / n)
    s["avg_latency"] = round(sum(r.get("latency_s") or 0 for r in rows) / n, 1)
    s["total_tokens"] = sum((r.get("prompt_tokens") or 0) + (r.get("output_tokens") or 0) for r in rows)
    return s


def main(argv: list[str]) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    n = 5
    models = []
    args = iter(argv)
    for a in args:
        if a == "--n":
            n = int(next(args))
        else:
            models.append(a)
    if not models:
        sys.exit("usage: python -m benchmark.bakeoff provider:model [provider:model ...] [--n 5]")
    case_ids = pick_cases(n)
    print(f"cases ({len(case_ids)}): {', '.join(case_ids)}")
    print(f"calls: {len(case_ids) * 2} per model (positive + negative diff for each case)\n")
    results = []
    for m in models:
        print(f"== {m}")
        t0 = time.time()
        res = run_model(m, case_ids)
        print("\n".join("  " + d for d in res["detail"]))
        for e in res["errors"]:
            print("  ! " + e)
        print(f"  ({time.time() - t0:.0f}s)\n")
        results.append(res)
    print("SUMMARY (location-only scoring, a proxy until the judge exists)")
    print(f"{'model':36s} {'valid':>7s} {'found':>7s} {'noise':>6s} {'quiet-on-fix':>13s} "
          f"{'out tok':>8s} {'secs':>5s} {'total tok':>10s}")
    for r in results:
        valid = f"{r['pos_ok'] + r['neg_ok']}/{2 * len(case_ids)}"
        found = f"{r['pos_hit']}/{len(case_ids)}"
        quiet = f"{r['neg_ok'] - r['neg_fp_cases']}/{len(case_ids)}"
        print(f"{r['model']:36s} {valid:>7s} {found:>7s} {r['pos_extra']:>6d} {quiet:>13s} "
              f"{r['avg_output']:>8d} {r['avg_latency']:>5.1f} {r['total_tokens']:>10d}")
    out = Path(__file__).resolve().parent.parent / "data" / "bakeoff"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_{int(time.time())}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
