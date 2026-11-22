"""Ground-truth location tests: change groups inside mixed hunks, and meta.json in sync with diffs."""
import json

from benchmark.build_cases import CASES, hunk_location, hunk_locations, parse_patch
from benchmark.refresh_locations import locations_from_diff

HEADER = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"


def locs(body):
    out = []
    for f in parse_patch(HEADER + body):
        for h in f["hunks"]:
            out.extend(hunk_locations(f["path"], h))
    return out


def test_missing_check_far_from_the_removal_is_kept():
    body = ("@@ -10,14 +10,15 @@\n c1\n c2\n-bad = 1\n+good = 1\n c3\n c4\n c5\n c6\n c7\n c8\n"
            "+if x is None:\n+    return\n c9\n c10\n")
    assert locs(body) == [
        {"file": "a.py", "start": 12, "end": 12, "anchor_only": False},
        {"file": "a.py", "start": 18, "end": 18, "anchor_only": True},
    ]


def test_insertion_inside_the_removed_range_adds_nothing():
    body = "@@ -10,6 +10,7 @@\n c\n-bad1\n+good1\n mid\n+extra\n-bad2\n+good2\n"
    assert locs(body) == [{"file": "a.py", "start": 11, "end": 13, "anchor_only": False}]


def test_replacement_only_hunk_is_unchanged_from_hunk_location():
    body = "@@ -10,5 +10,5 @@\n ctx1\n ctx2\n-bad = 1\n+good = 1\n ctx3\n ctx4\n"
    f = parse_patch(HEADER + body)[0]
    assert locs(body) == [hunk_location("a.py", f["hunks"][0])]


def test_two_pure_insertions_in_one_hunk_give_two_anchors():
    body = "@@ -40,5 +40,7 @@\n a\n+new1\n b\n c\n d\n+new2\n e\n"
    assert [(l["start"], l["anchor_only"]) for l in locs(body)] == [(40, True), (43, True)]


def test_comment_only_insertion_group_is_ignored():
    body = ("@@ -10,12 +10,13 @@\n c1\n-bad = 1\n+good = 1\n c2\n c3\n c4\n c5\n c6\n c7\n"
            "+# just a comment\n c8\n")
    assert locs(body) == [{"file": "a.py", "start": 11, "end": 11, "anchor_only": False}]


def test_comment_only_hunk_still_gives_nothing():
    assert locs("@@ -5,3 +5,2 @@\n x\n-# old comment\n y\n") == []


def test_stored_meta_matches_the_stored_diffs():
    """Fails if meta.json bug_locations drift from negative.diff: run `python -m benchmark.refresh_locations --write`."""
    stale = []
    for d in sorted(CASES.iterdir()):
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            diff = (d / "negative.diff").read_text(encoding="utf-8")
            if locations_from_diff(diff) != meta["bug_locations"]:
                stale.append(d.name)
    assert not stale, stale