"""Build review cases from verified BugsInPy bugs (Phase 2).

For every bug that reproduces (and passes the quality filter) this writes
benchmark/cases/<project>-<id>/ with:
  positive.diff   fixed -> buggy   (looks like someone introduced the bug; reviewer should flag it)
  negative.diff   buggy -> fixed   (the correct change; anything flagged is a candidate false positive)
  meta.json       ground truth: bug location (buggy-side lines), commits, files, test info

File contents used as review context are NOT stored here. The harness reads them with
`git show <commit>:<path>` from data/repos/<project> (buggy commit for the positive case,
fixed commit for the negative case). The BugsInPy test is ground truth only and must never
be shown to the reviewer or the repro generator.

Usage:  python -m benchmark.build_cases
"""
import json
import re
import subprocess

from benchmark.verify_repro import BUGSINPY, REPOS, RESULTS, ROOT, WORK, read_kv

CASES = ROOT / "benchmark" / "cases"

# Quality filter: the test fails only because it patches a symbol the fix introduced.
EXCLUDE_MARKER = "does not have the attribute"
# Test and test-support files are never part of a review case.
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")


def parse_patch(text):
    """Parse a unified diff into [{path, hunks: [{old_start, old_count, lines}]}]."""
    files, cur, hunk = [], None, None
    for line in text.split("\n"):
        line = line.rstrip("\r")
        m = FILE_RE.match(line)
        if m:
            cur = {"path": m.group(2), "hunks": []}
            files.append(cur)
            hunk = None
            continue
        m = HUNK_RE.match(line)
        if m and cur is not None:
            hunk = {
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2)) if m.group(2) is not None else 1,
                "lines": [],
            }
            cur["hunks"].append(hunk)
            continue
        if hunk is not None and line[:1] in ("+", "-", " ", "\\", ""):
            hunk["lines"].append(line)
    return files


def hunk_location(path, hunk):
    """Buggy-side location of one hunk of a buggy->fixed diff.

    Removed lines are buggy code, so the range is min..max of removed line numbers.
    A pure addition (code the buggy version is missing) has no buggy lines, so it is
    anchored at the line after which the code is missing.
    """
    changed = [l[1:].strip() for l in hunk["lines"] if l[:1] in ("+", "-")]
    if all(c == "" or c.startswith("#") for c in changed):
        return None  # comment/whitespace-only hunk: part of the diff, not a bug location
    old = hunk["old_start"] if hunk["old_count"] > 0 else hunk["old_start"] + 1
    removed, anchor = [], None
    for ln in hunk["lines"]:
        if ln.startswith("-"):
            removed.append(old)
            old += 1
        elif ln.startswith("+"):
            if anchor is None:
                anchor = old - 1
        elif ln.startswith("\\"):
            continue
        else:
            old += 1
    if removed:
        return {"file": path, "start": min(removed), "end": max(removed), "anchor_only": False}
    if anchor is not None:
        return {"file": path, "start": anchor, "end": anchor, "anchor_only": True}
    return None


def signature(files):
    """Per file: (sorted removed old-line numbers, number of added lines). Used to check
    our git diff against BugsInPy's own bug_patch.txt."""
    sig = {}
    for f in files:
        removed, added = [], 0
        for h in f["hunks"]:
            old = h["old_start"] if h["old_count"] > 0 else h["old_start"] + 1
            for ln in h["lines"]:
                if ln.startswith("-"):
                    removed.append(old)
                    old += 1
                elif ln.startswith("+"):
                    added += 1
                elif ln.startswith("\\"):
                    continue
                else:
                    old += 1
        sig[f["path"]] = (sorted(removed), added)
    return sig


def git_diff(repo, a, b, paths):
    r = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(repo), "diff", "--no-color", "-U3", a, b, "--", *paths],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    return r.stdout


def changed_lines(text):
    return sum(1 for l in text.split("\n") if l[:1] in "+-" and not l.startswith(("+++", "---")))


def build(res):
    project, bug_id = res["project"], res["bug_id"]
    tag = f"{project}-{bug_id}"
    bug_dir = BUGSINPY / project / "bugs" / str(bug_id)
    info = read_kv(bug_dir / "bug.info")
    proj_info = read_kv(BUGSINPY / project / "project.info")

    log = WORK / f"{tag}-out" / "buggy.log"
    log_text = log.read_text(errors="replace") if log.exists() else res.get("buggy_tail", "")
    if EXCLUDE_MARKER in log_text:
        return None, "test fails on a symbol the fix introduced"

    patch_files = parse_patch((bug_dir / "bug_patch.txt").read_text(encoding="utf-8", errors="replace"))
    src_files = [f for f in patch_files if not TEST_PATH_RE.search(f["path"])]
    if not src_files:
        return None, "patch only touches test files"
    paths = [f["path"] for f in src_files]

    repo = REPOS / project
    buggy_c, fixed_c = info["buggy_commit_id"], info["fixed_commit_id"]
    negative = git_diff(repo, buggy_c, fixed_c, paths)
    positive = git_diff(repo, fixed_c, buggy_c, paths)
    if not negative.strip() or not positive.strip():
        return None, "empty diff between commits"

    locations = []
    for f in parse_patch(negative):
        for h in f["hunks"]:
            loc = hunk_location(f["path"], h)
            if loc:
                locations.append(loc)

    if not locations:
        return None, "no code location (only comment/whitespace changes)"

    meta = {
        "case_id": tag, "project": project, "bug_id": int(bug_id), "split": res.get("split", "dev"),
        "python_version": info.get("python_version"), "github_url": proj_info.get("github_url"),
        "buggy_commit": buggy_c, "fixed_commit": fixed_c,
        "files": paths,
        "bug_locations": locations,
        "patch_stats": {
            "files": len(paths),
            "hunks": sum(len(f["hunks"]) for f in parse_patch(negative)),
            "changed_lines": changed_lines(negative),
        },
        "matches_bugsinpy_patch": signature(parse_patch(negative)) == signature(src_files),
        "category": None,  # to be assigned (logic / security / error_handling / ...)
        "ground_truth_test": {"file": info.get("test_file"), "never_show_to_agent": True},
    }
    out = CASES / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "positive.diff").write_text(positive, encoding="utf-8", newline="\n")
    (out / "negative.diff").write_text(negative, encoding="utf-8", newline="\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return meta, None


def main():
    built, skipped = [], []
    for p in sorted(RESULTS.glob("*.json")):
        res = json.loads(p.read_text(encoding="utf-8"))
        if not res.get("reproduces"):
            skipped.append((p.stem, "did not reproduce"))
            continue
        try:
            meta, why = build(res)
        except Exception as e:  # noqa: BLE001
            meta, why = None, f"error: {e}"
        if meta:
            built.append(meta)
        else:
            skipped.append((p.stem, why))

    CASES.mkdir(parents=True, exist_ok=True)
    (CASES / "index.json").write_text(
        json.dumps([{"case_id": m["case_id"], "split": m["split"]} for m in built], indent=2),
        encoding="utf-8", newline="\n",
    )
    print(f"built {len(built)} cases, skipped {len(skipped)}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    bad = [m["case_id"] for m in built if not m["matches_bugsinpy_patch"]]
    anchors = [m["case_id"] for m in built if any(l["anchor_only"] for l in m["bug_locations"])]
    print(f"patch mismatches vs BugsInPy: {bad or 'none'}")
    print(f"cases with an anchor-only (missing code) location: {len(anchors)}")


if __name__ == "__main__":
    main()
