"""Hand-label validation of the judge (docs/evaluation.md, "Judge rules").

sample     writes a blinded reading file and a CSV to fill in: up to N findings the judge said yes
           to and up to N it said no to, shuffled, WITHOUT the verdicts (so you are not anchored),
           plus negative-run findings to check for real latent issues.
agreement  compares your filled CSV with the judge: accuracy with a Wilson interval, Cohen's kappa,
           the confusion table and every disagreement.
latent     counts how many sampled negative-run findings you marked as real issues.

Usage (repo root, venv active):
  python -m benchmark.handlabel sample data/runs/pilot.jsonl [--n 15] [--seed 0]
  python -m benchmark.handlabel agreement
  python -m benchmark.handlabel latent
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Optional

from benchmark import judge, labeling, metrics as mt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "runs" / "handlabel"


def sample_items(records: dict, cases: dict, verdicts: dict, n_per_class: int = 15, seed: int = 0,
                 n_negative: int = 10) -> dict:
    """{'judged': [(item, verdict)...] shuffled, 'negative': [item...]}; items are queue-style dicts."""
    rng = random.Random(seed)
    yes, no = [], []
    for rid, rec in sorted(records.items()):
        if rec.get("status") != "ok" or rec["case_id"] not in cases or rec["kind"] != "positive":
            continue
        for f in rec["findings"]:
            v = verdicts.get(f["id"])
            if v:
                (yes if v == "yes" else no).append({"finding_id": f["id"], "case_id": rec["case_id"], "finding": f})
    rng.shuffle(yes)
    rng.shuffle(no)
    judged = [(i, "yes") for i in yes[:n_per_class]] + [(i, "no") for i in no[:n_per_class]]
    rng.shuffle(judged)
    neg = [{"finding_id": f["id"], "case_id": rec["case_id"], "finding": f}
           for rid, rec in sorted(records.items())
           if rec.get("status") == "ok" and rec["case_id"] in cases and rec["kind"] == "negative"
           for f in rec["findings"]]
    rng.shuffle(neg)
    return {"judged": judged, "negative": neg[:n_negative], "available": (len(yes), len(no), len(neg))}


def _block(item: dict, cases_dir: Path) -> list[str]:
    f = item["finding"]
    patch = (cases_dir / item["case_id"] / "negative.diff").read_text(encoding="utf-8").rstrip()
    return [f"### {item['finding_id']}", f"Finding at {f['file']} line {f['line']}:", f"> {f['message']}",
            f"> expected failure: {f.get('expected_failure') or 'none given'}", "",
            "Patch (buggy -> fixed):", "```diff", patch, "```", ""]


def write_files(sampled: dict, out_dir: Path, cases_dir: Optional[Path] = None) -> tuple[Path, Path, Path]:
    cases_dir = cases_dir or labeling.CASES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    md, csv_main, csv_neg = out_dir / "handlabel.md", out_dir / "handlabel.csv", out_dir / "negatives.csv"
    lines = ["# Hand labels", "",
             "Part 1: does the finding describe the defect this patch fixes? Same underlying problem, same place,",
             "consistent explanation; a finding that names the right defect with a slightly different symptom is a yes.",
             "The reviewer saw the change reversed, so its wording may say 'removed' where the patch adds code.",
             "Fill `human_label` (yes or no) in handlabel.csv.", ""]
    for item, _ in sampled["judged"]:
        lines += _block(item, cases_dir)
    lines += ["", "# Part 2: negative-run findings", "",
              "The reviewer was shown the FIXED code as a change. Is the finding a real defect or risk in that code",
              "(not a restatement of the fix)? Fill `real_issue` (yes, no or unsure) in negatives.csv.", ""]
    for item in sampled["negative"]:
        lines += _block(item, cases_dir)
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    for path, col, rows in ((csv_main, "human_label", [i for i, _ in sampled["judged"]]),
                            (csv_neg, "real_issue", sampled["negative"])):
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["finding_id", col])
            for i in rows:
                w.writerow([i["finding_id"], ""])
    return md, csv_main, csv_neg


def read_labels(path: Path, column: str, allowed: tuple[str, ...]) -> dict[str, str]:
    out = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            v = (row.get(column) or "").strip().lower()
            if not v:
                continue
            if v not in allowed:
                raise ValueError(f"{row['finding_id']}: {column} must be one of {allowed}, got {v!r}")
            out[row["finding_id"]] = v
    return out


def cohens_kappa(pairs: list[tuple[str, str]]) -> Optional[float]:
    """Kappa for two yes/no raters. None when chance agreement is already 1 (no variation)."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in ("yes", "no"))
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(human: dict[str, str], verdicts: dict[str, str]) -> dict:
    pairs = [(verdicts[k], h) for k, h in sorted(human.items()) if k in verdicts]
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    conf = {(j, h): sum(1 for p in pairs if p == (j, h)) for j in ("yes", "no") for h in ("yes", "no")}
    return {"n": n, "agree": agree, "accuracy": agree / n if n else None,
            "ci": mt.wilson(agree, n), "kappa": cohens_kappa(pairs), "confusion": conf,
            "disagreements": [(k, verdicts[k], h) for k, h in sorted(human.items()) if k in verdicts and verdicts[k] != h],
            "missing": sorted(k for k in human if k not in verdicts)}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Hand-label validation of the judge.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("runs", type=Path)
    s.add_argument("--labels", type=Path, default=judge.LABELS_PATH)
    s.add_argument("--n", type=int, default=15, help="findings per class (judge yes / judge no)")
    s.add_argument("--negatives", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--split", default="dev")
    s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    g = sub.add_parser("agreement")
    g.add_argument("--csv", type=Path, default=OUT_DIR / "handlabel.csv")
    g.add_argument("--labels", type=Path, default=judge.LABELS_PATH)
    t = sub.add_parser("latent")
    t.add_argument("--csv", type=Path, default=OUT_DIR / "negatives.csv")
    a = ap.parse_args(argv)

    if a.cmd == "sample":
        sampled = sample_items(labeling.load_latest_records(a.runs), labeling.load_cases(split=a.split),
                               judge.load_verdicts(a.labels), a.n, a.seed, a.negatives)
        ny, nn, nneg = sampled["available"]
        md, c1, c2 = write_files(sampled, a.out_dir)
        print(f"judged yes available {ny}, judged no available {nn} (asked for {a.n} of each); "
              f"sampled {len(sampled['judged'])} judged + {len(sampled['negative'])} negative-run findings of {nneg}")
        if min(ny, nn) < a.n:
            print("NOTE: fewer than the requested number in a class; the protocol asks for 15 yes + 15 no, "
                  "so repeat on the full dev run.")
        print(f"read:  {md}\nfill:  {c1}  (column human_label: yes/no)\nfill:  {c2}  (column real_issue: yes/no/unsure)")
    elif a.cmd == "agreement":
        r = agreement(read_labels(a.csv, "human_label", ("yes", "no")), judge.load_verdicts(a.labels))
        if not r["n"]:
            sys.exit("no filled-in rows that the judge also labeled yet")
        lo, hi = r["ci"]
        print(f"labeled {r['n']}   agreement {r['agree']}/{r['n']} = {r['accuracy']:.2f} [Wilson {lo:.2f}, {hi:.2f}]"
              f"   Cohen's kappa {'n/a' if r['kappa'] is None else format(r['kappa'], '.2f')}")
        c = r["confusion"]
        print("             human yes  human no\n"
              f"  judge yes  {c[('yes', 'yes')]:9d}  {c[('yes', 'no')]:8d}\n  judge no   {c[('no', 'yes')]:9d}  {c[('no', 'no')]:8d}")
        for fid, j, h in r["disagreements"]:
            print(f"  DISAGREE {fid}: judge {j}, you {h}")
        if r["missing"]:
            print(f"  not judged (ignored): {', '.join(r['missing'])}")
    else:
        v = read_labels(a.csv, "real_issue", ("yes", "no", "unsure"))
        n = len(v)
        if not n:
            sys.exit("no filled-in rows yet")
        from collections import Counter
        c = Counter(v.values())
        print(f"negative-run findings checked: {n}   real issue {c['yes']}, not an issue {c['no']}, unsure {c['unsure']}")


if __name__ == "__main__":
    main(sys.argv[1:])