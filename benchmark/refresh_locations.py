"""Recompute bug_locations in each benchmark/cases/*/meta.json from its stored negative.diff.

Needs no BugsInPy checkout and no Docker: negative.diff is the buggy->fixed diff the locations
come from. Only bug_locations is touched. Default is a dry run; add --write to save.

  python -m benchmark.refresh_locations            # show what would change
  python -m benchmark.refresh_locations --write    # rewrite the changed meta.json files
"""
import json
import sys

from benchmark.build_cases import CASES, hunk_locations, parse_patch


def locations_from_diff(diff_text: str) -> list[dict]:
    out = []
    for f in parse_patch(diff_text):
        for h in f["hunks"]:
            out.extend(hunk_locations(f["path"], h))
    return out


def fmt(locs: list[dict]) -> str:
    return ", ".join(f"{l['file']}:{l['start']}" + (f"-{l['end']}" if l["end"] != l["start"] else "")
                     + ("(anchor)" if l["anchor_only"] else "") for l in locs)


def main(argv: list[str]) -> None:
    write = "--write" in argv
    changed = 0
    for d in sorted(CASES.iterdir()):
        mp, nd = d / "meta.json", d / "negative.diff"
        if not (mp.exists() and nd.exists()):
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        new = locations_from_diff(nd.read_text(encoding="utf-8"))
        if new == meta["bug_locations"]:
            continue
        changed += 1
        print(f"{d.name}\n   was: {fmt(meta['bug_locations'])}\n   now: {fmt(new)}")
        if write:
            meta["bug_locations"] = new
            mp.write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    print(f"\n{changed} case(s) " + ("rewritten" if write else "would change (dry run, nothing written)"))


if __name__ == "__main__":
    main(sys.argv[1:])