from benchmark.build_cases import TEST_PATH_RE, hunk_location, parse_patch


def locs(diff):
    out = []
    for f in parse_patch(diff):
        for h in f["hunks"]:
            out.append(hunk_location(f["path"], h))
    return out


HEADER = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"


def test_replacement_points_at_removed_buggy_line():
    d = HEADER + "@@ -10,5 +10,5 @@\n ctx1\n ctx2\n-bad = 1\n+good = 1\n ctx3\n ctx4\n"
    assert locs(d) == [{"file": "a.py", "start": 12, "end": 12, "anchor_only": False}]


def test_removed_lines_with_context_between_give_a_range():
    d = HEADER + "@@ -10,5 +10,5 @@\n ctx\n-bad1\n+good1\n mid\n-bad2\n+good2\n"
    assert locs(d) == [{"file": "a.py", "start": 11, "end": 13, "anchor_only": False}]


def test_pure_addition_is_anchored_after_the_previous_line():
    d = HEADER + "@@ -40,3 +40,4 @@\n a\n b\n+new = 1\n c\n"
    assert locs(d) == [{"file": "a.py", "start": 41, "end": 41, "anchor_only": True}]


def test_zero_count_hunk_is_anchored_at_old_start():
    d = HEADER + "@@ -7,0 +8,2 @@\n+x = 1\n+y = 2\n"
    assert locs(d) == [{"file": "a.py", "start": 7, "end": 7, "anchor_only": True}]


def test_comment_only_hunk_is_not_a_bug_location():
    d = HEADER + "@@ -5,3 +5,2 @@\n x\n-# old comment\n y\n"
    assert locs(d) == [None]


def test_test_paths_are_recognised():
    for p in ("tornado/test/runtests.py", "tests/rules/test_x.py", "pkg/foo_test.py"):
        assert TEST_PATH_RE.search(p), p
    for p in ("tornado/http1connection.py", "thefuck/rules/pip_unknown_command.py", "httpie/client.py"):
        assert not TEST_PATH_RE.search(p), p
