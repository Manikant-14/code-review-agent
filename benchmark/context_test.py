"""Step 3: does bounded context help the reviewer? Same 5 bake-off cases, with and without it.

Arm A (no context) uses the exact prompt, system text and schema of the bake-off, so its calls are
cache hits and cost nothing. Arm B adds the enclosing function of each hunk (src/review_input.py).
Location-only scoring, n = 5: a screening check, not a ranking. Do not tune prompts on it.

Usage (repo root, venv active):
  python -m benchmark.context_test --dry-run      # no API calls: sizes and cache hits only
  python -m benchmark.context_test                # real run (10 new Groq calls, ~5 minutes)
"""
import json
import sys
import time

from benchmark import bakeoff
from benchmark.bakeoff import CASES_DIR, Review, is_hit
from src import config, llm
from src.review_input import build_review_input

CONTEXT_SYSTEM = bakeoff.SYSTEM + (
    "\nAfter the diff you may get 'context' blocks: the surrounding code of the new version, with "
    "line numbers. Use them only to understand the change; report defects in the changed lines."
)
ARMS = {"A no-context": (False, bakeoff.SYSTEM), "B context": (True, CONTEXT_SYSTEM)}


def _cached(prompt: str, system: str) -> bool:
    key = llm._key(config.MODELS["reviewer"], config.TEMPERATURES["reviewer"], system, prompt, Review, 0)
    return (llm.CACHE_DIR / key[:2] / f"{key}.json").exists()


def main(argv: list[str]) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    dry = "--dry-run" in argv
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 5
    model = config.MODELS["reviewer"]
    config.RPM_LIMITS[model] = 2  # Groq's limit is 8K tokens/min: bigger prompts need a slower pace
    case_ids = bakeoff.pick_cases(n)
    print(f"cases: {', '.join(case_ids)}\nreviewer: {model}\n")
    offset = llm.LOG_PATH.stat().st_size if llm.LOG_PATH.exists() else 0
    stats = {a: {"ok": 0, "hit": 0, "extra": 0, "neg_ok": 0, "neg_fp": 0, "est": [], "ctx": 0} for a in ARMS}
    new_calls = 0
    for cid in case_ids:
        meta = json.loads((CASES_DIR / cid / "meta.json").read_text(encoding="utf-8"))
        for kind in ("positive", "negative"):
            for arm, (with_ctx, system) in ARMS.items():
                ri = build_review_input(cid, kind, with_context=with_ctx)
                st = stats[arm]
                st["est"].append(ri.est_tokens)
                st["ctx"] += ri.with_context
                cached = _cached(ri.text, system)
                new_calls += not cached
                tag = f"{cid:11s} {kind:8s} {arm:12s} ~{ri.est_tokens:5d} tok {'cached' if cached else 'NEW   '}"
                if dry:
                    print(tag + (f"  ctx={ri.context_lines} lines" if with_ctx else "") + (f"  {ri.notes}" if ri.notes else ""))
                    continue
                try:
                    fs = llm.call_llm("reviewer", ri.text, system=system, schema=Review).findings
                except llm.DailyQuotaExhausted as e:
                    sys.exit(f"daily quota exhausted: {str(e)[:100]}")
                except Exception as e:  # noqa: BLE001
                    print(f"{tag}  ERROR {type(e).__name__}: {str(e)[:90]}")
                    continue
                if kind == "positive":
                    hits = [f for f in fs if is_hit(f, meta["bug_locations"], config.LOCATION_MATCH_WINDOW)]
                    st["ok"] += 1
                    st["hit"] += bool(hits)
                    st["extra"] += len(fs) - len(hits)
                    print(f"{tag}  {'HIT ' if hits else 'miss'} findings={len(fs)}" + bakeoff._msgs(fs))
                else:
                    st["neg_ok"] += 1
                    st["neg_fp"] += bool(fs)
                    print(f"{tag}  {'quiet' if not fs else 'FALSE POSITIVE x' + str(len(fs))}" + bakeoff._msgs(fs))
    if dry:
        print(f"\ndry run: {new_calls} calls would be NEW (rest are cached)")
        return
    rows = bakeoff._log_rows_since(offset, model)
    print("\nSUMMARY (location-only, n small: screening only)")
    print(f"{'arm':14s} {'found':>6s} {'noise':>6s} {'quiet-on-fix':>13s} {'est tok (max)':>14s}")
    for arm, st in stats.items():
        print(f"{arm:14s} {st['hit']}/{st['ok']:<4d} {st['extra']:>6d} {st['neg_ok'] - st['neg_fp']}/{st['neg_ok']:<11d} {max(st['est']):>14d}")
    if rows:
        print(f"\nnew calls: {len(rows)}, provider-reported prompt tokens avg "
              f"{sum(r.get('prompt_tokens') or 0 for r in rows) / len(rows):.0f}, "
              f"output avg {sum(r.get('output_tokens') or 0 for r in rows) / len(rows):.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])