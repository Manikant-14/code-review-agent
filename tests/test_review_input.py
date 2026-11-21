"""Tests for src/review_input.py (no git, no network: sources are injected)."""
import json

import pytest

from src import config, review_input as ri

SOURCE = "\n".join([
    "import os",                      # 1
    "",                               # 2
    "class A:",                       # 3
    "    def f(self, x):",            # 4
    "        y = x + 1",              # 5
    "        return y",               # 6
    "",                               # 7
    "    def g(self, x):",            # 8
    "        z = x * 2",              # 9
    "        return z",               # 10
    "",                               # 11
    "def top(a):",                    # 12
    "    return a",                   # 13
]) + "\n"

DIFF = (
    "diff --git a/m.py b/m.py\n"
    "index 111..222 100644\n"
    "--- a/m.py\n"
    "+++ b/m.py\n"
    "@@ -8,3 +8,3 @@ class A:\n"
    "     def g(self, x):\n"
    "-        z = x + 2\n"
    "+        z = x * 2\n"
    "         return z\n"
)


def make_case(tmp_path, diff=DIFF, cid="proj-1"):
    d = tmp_path / cid
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(
        {"project": "proj", "buggy_commit": "bbbbbbbb", "fixed_commit": "ffffffff"}), encoding="utf-8")
    (d / "positive.diff").write_text(diff, encoding="utf-8")
    (d / "negative.diff").write_text(diff, encoding="utf-8")
    return tmp_path


def test_annotate_diff_numbers_lines_and_drops_index():
    out = ri.annotate_diff(DIFF)
    assert "index 111" not in out and "diff --git" not in out
    assert "    -|         z = x + 2" in out
    assert "    9+|         z = x * 2" in out
    assert "=== m.py" in out


def test_function_spans_and_innermost():
    spans = ri.function_spans(SOURCE)
    assert (4, 6) in spans and (8, 10) in spans and (12, 13) in spans
    assert ri.innermost(spans, 9) == (8, 10)
    assert ri.innermost(spans, 1) is None


def test_function_spans_include_decorators_and_survive_bad_source():
    src = "@deco\ndef f():\n    pass\n"
    assert ri.function_spans(src) == [(1, 3)]
    assert ri.function_spans("def broken(:\n") == []


def test_context_is_enclosing_function(tmp_path):
    cases = make_case(tmp_path)
    x = ri.build_review_input("proj-1", "positive", cases_dir=cases, read_file=lambda p: SOURCE)
    assert x.with_context and x.context_lines == 3
    assert "context: m.py (new version, lines 8-10)" in x.text
    assert "    8 |     def g(self, x):" in x.text
    assert "def f(self" not in x.text.split("=== context")[1]


def test_no_context_prompt_is_just_head_plus_diff(tmp_path):
    cases = make_case(tmp_path)
    x = ri.build_review_input("proj-1", "positive", with_context=False, cases_dir=cases)
    assert x.text == ri.PROMPT_HEAD + ri.annotate_diff(DIFF)
    assert not x.with_context


def test_missing_source_skips_context_with_note(tmp_path):
    cases = make_case(tmp_path)
    x = ri.build_review_input("proj-1", "negative", cases_dir=cases, read_file=lambda p: None)
    assert not x.with_context and "context skipped" in x.notes[0]


def test_commit_choice_follows_kind(tmp_path):
    cases = make_case(tmp_path)
    seen = []
    orig = ri.git_show
    ri.git_show = lambda project, commit, path, repos_dir=None: seen.append(commit) or SOURCE
    try:
        ri.build_review_input("proj-1", "positive", cases_dir=cases)
        ri.build_review_input("proj-1", "negative", cases_dir=cases)
    finally:
        ri.git_show = orig
    assert seen == ["bbbbbbbb", "ffffffff"]


def test_long_function_is_clipped_around_the_hunk():
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    src = "def big():\n" + body + "\n"
    rs = ri.context_ranges(src, [(150, 152)], max_func_lines=40, radius=5)
    assert len(rs) == 1
    s, e = rs[0]
    assert e - s + 1 <= 40 and s <= 150 and e >= 152


def test_fallback_window_when_no_function_encloses_hunk():
    src = "\n".join(f"v{i} = {i}" for i in range(100)) + "\n"
    assert ri.context_ranges(src, [(50, 50)], 80, 7) == [(43, 57)]


def test_window_is_clamped_to_file_bounds():
    src = "a = 1\nb = 2\n"
    assert ri.context_ranges(src, [(1, 2)], 80, 20) == [(1, 2)]


def test_overlapping_ranges_merge():
    assert ri._merge([(1, 5), (4, 9), (11, 12), (13, 14)]) == [(1, 9), (11, 14)]


def test_budget_shrinks_then_drops_context(tmp_path, monkeypatch):
    cases = make_case(tmp_path)
    big = "def g(x):\n" + "\n".join(f"    v{i} = {i}" for i in range(400)) + "\n"
    diff = ("diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -200,1 +200,1 @@\n"
            "-    v198 = 0\n+    v198 = 198\n")
    cases = make_case(tmp_path, diff=diff, cid="proj-2")
    monkeypatch.setattr(config, "REVIEW_MAX_INPUT_TOKENS", 400)
    x = ri.build_review_input("proj-2", "positive", cases_dir=cases, read_file=lambda p: big)
    assert x.with_context and x.est_tokens <= 400
    assert any("shrunk" in n for n in x.notes)
    monkeypatch.setattr(config, "REVIEW_MAX_INPUT_TOKENS", 5)
    y = ri.build_review_input("proj-2", "positive", cases_dir=cases, read_file=lambda p: big)
    assert not y.with_context and y.over_budget


def test_bad_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        ri.build_review_input("x", "maybe", cases_dir=tmp_path)