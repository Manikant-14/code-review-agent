"""Quick look at a reviewer JSONL (step 5 pilot). Location-only: the real harness is step 6.

Known gap (STATUS finding 2): thefuck-10, thefuck-24, thefuck-28 and httpie-5 have incomplete
ground-truth locations, so their misses may be scorer misses. Marked with * below.

Usage: python -m benchmark.pilot_summary [data/runs/pilot.jsonl]
"""
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmark.bakeoff import CASES_DIR, is_hit
from src import config

GAP_CASES = {"thefuck-10", "thefuck-24", "thefuck-28", "httpie-5"}


def load_latest(path: Path) -> dict:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec["run_id"]] = rec  # a later line (a retry) replaces an earlier error
    return latest


def summarise(latest: dict, cases_dir: Path = CASES_DIR) -> dict:
    out = {"cases": {}, "errors": [], "hit_conf": [], "other_conf": []}
    for rid, rec in sorted(latest.items()):
        if rec["status"] != "ok":
            out["errors"].append((rid, rec.get("error")))
            continue
        cid, kind = rec["case_id"], rec["kind"]
        row = out["cases"].setdefault(cid, {})
        fs = rec["findings"]
        if kind == "positive":
            locs = json.loads((cases_dir / cid / "meta.json").read_text(encoding="utf-8"))["bug_locations"]
            hits = [f for f in fs if is_hit(SimpleNamespace(**f), locs, config.LOCATION_MATCH_WINDOW)]
            row["pos"] = (bool(hits), len(fs))
            out["hit_conf"] += [f["self_confidence"] for f in hits]
            out["other_conf"] += [f["self_confidence"] for f in fs if f not in hits]
        else:
            row["neg"] = len(fs)
            out["other_conf"] += [f["self_confidence"] for f in fs]
        row.setdefault("est", []).append(rec["est_tokens"])
    return out


def main(argv: list[str]) -> None:
    path = Path(argv[0]) if argv else Path("data/runs/pilot.jsonl")
    s = summarise(load_latest(path))
    print(f"{'case':12s} {'positive run':22s} {'negative run'}")
    pos = [r["pos"] for r in s["cases"].values() if "pos" in r]
    neg = [r["neg"] for r in s["cases"].values() if "neg" in r]
    for cid, r in s["cases"].items():
        p = "-" if "pos" not in r else f"{'HIT ' if r['pos'][0] else 'miss'} ({r['pos'][1]} findings)"
        n = "-" if "neg" not in r else ("quiet" if r["neg"] == 0 else f"{r['neg']} finding(s) on the fix")
        print(f"{cid + ('*' if cid in GAP_CASES else ''):12s} {p:22s} {n}")
    print(f"\nlocation recall   {sum(h for h, _ in pos)}/{len(pos)} positive runs"
          f"   (* = ground-truth gap, may be a scorer miss)")
    print(f"fix-flag rate     {sum(1 for n in neg if n)}/{len(neg)} negative runs had a finding")
    if pos:
        print(f"findings per run  positive {statistics.mean(n for _, n in pos):.2f}"
              f"   negative {statistics.mean(neg) if neg else 0:.2f}")
    if s["hit_conf"] and s["other_conf"]:
        print(f"self_confidence   location hits {statistics.mean(s['hit_conf']):.2f} (n={len(s['hit_conf'])})"
              f"   all other findings {statistics.mean(s['other_conf']):.2f} (n={len(s['other_conf'])})")
    if s["errors"]:
        print(f"\n{len(s['errors'])} run(s) still in error:")
        for rid, e in s["errors"]:
            print(f"  {rid}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])