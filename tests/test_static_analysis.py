import json
import subprocess

import pytest

from src import config, static_analysis as sa
from src.state import Finding

needs_ruff = pytest.mark.skipif(sa.ruff_version() is None, reason="ruff does not run here")


def make_finding(fid="r:f0", file="a.py", line=3, **kw):
    base = dict(id=fid, file=file, line=line, category="logic", severity="medium", message="m")
    base.update(kw)
    return Finding(**base)


def diff_adding(path, old, new):
    """A real unified diff from old text to new text (one file)."""
    import difflib
    body = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        f"a/{path}", f"b/{path}", n=2))
    return f"diff --git a/{path} b/{path}\n{body}"


# ---- pure helpers

def test_added_lines_new_side_numbers_only():
    d = diff_adding("a.py", "x = 1\ny = 2\nz = 3\n", "x = 1\nimport os\ny = 2\nz = 4\n")
    assert sa.added_lines(d) == {"a.py": {2, 4}}


def test_added_lines_skips_non_python_and_deleted_files():
    d = diff_adding("README.md", "a\n", "a\nb\n")
    assert sa.added_lines(d) == {}
    deleted = ("diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n"
               "@@ -1,2 +0,0 @@\n-a = 1\n-b = 2\n")
    assert sa.added_lines(deleted) == {}


def test_added_lines_pure_deletion_has_no_added_lines():
    d = diff_adding("a.py", "x = 1\ny = 2\n", "x = 1\n")
    assert sa.added_lines(d) == {}


def test_filter_changed_uses_row_span():
    issues = [{"row": 5, "end_row": 8}, {"row": 20, "end_row": 20}, {"row": 1, "end_row": 1}]
    kept = sa.filter_changed(issues, {7, 20})
    assert [i["row"] for i in kept] == [5, 20]


def test_overlaps_window_is_inclusive():
    issues = [{"row": 10, "end_row": 10}]
    assert sa.overlaps(issues, 13, 3) and sa.overlaps(issues, 7, 3)
    assert not sa.overlaps(issues, 14, 3) and not sa.overlaps(issues, 6, 3)


def test_overlaps_multiline_issue_counts():
    assert sa.overlaps([{"row": 10, "end_row": 20}], 22, 3)


def test_apply_static_is_one_or_unavailable_never_zero():
    issues = {"a.py": [{"row": 10, "end_row": 10}]}
    fs = [make_finding("r:f0", "a.py", 11), make_finding("r:f1", "a.py", 40), make_finding("r:f2", "b.py", 10)]
    out = sa.apply_static(fs, issues)
    assert [f.static_overlap for f in out] == [1.0, None, None]
    assert fs[0].static_overlap is None  # inputs are not mutated


# ---- ruff failures become StaticError, not a crash and not a 0.0

def test_run_ruff_blocked_binary_raises_static_error(monkeypatch):
    def boom(*a, **k):
        raise PermissionError("blocked by policy")
    monkeypatch.setattr(sa.subprocess, "run", boom)
    with pytest.raises(sa.StaticError):
        sa.run_ruff("x = 1\n", "a.py")
    assert sa.ruff_version() is None


def test_run_ruff_bad_exit_and_bad_json(monkeypatch):
    def fake(rc, out=b"", err=b""):
        return lambda *a, **k: subprocess.CompletedProcess(a, rc, out, err)
    monkeypatch.setattr(sa.subprocess, "run", fake(2, b"", b"boom"))
    with pytest.raises(sa.StaticError, match="exit 2"):
        sa.run_ruff("x = 1\n", "a.py")
    monkeypatch.setattr(sa.subprocess, "run", fake(1, b"not json"))
    with pytest.raises(sa.StaticError, match="unreadable"):
        sa.run_ruff("x = 1\n", "a.py")


# ---- real ruff

@needs_ruff
def test_run_ruff_finds_unused_import_and_clean_file():
    issues = sa.run_ruff("x = 1\nimport os\n", "a.py")
    assert [(i["code"], i["row"]) for i in issues] == [("F401", 2)]
    assert sa.run_ruff("x = 1\n", "a.py") == []


@needs_ruff
def test_style_rules_are_not_selected():
    # E701 (multiple statements on one line) and E731 (lambda assignment) are style rules: out of the set.
    assert sa.run_ruff("if True: pass\nf = lambda: 0\n", "a.py") == []


@needs_ruff
def test_run_ruff_ignores_project_config_and_handles_crlf_and_unicode():
    src = "# caf\u00e9\r\nimport os\r\n"
    assert [(i["code"], i["row"]) for i in sa.run_ruff(src, "a.py")] == [("F401", 2)]


def _state(old, new, findings, files=None, path="a.py"):
    return {"pr_ref": "r", "diff": diff_adding(path, old, new), "files": {path: new} if files is None else files,
            "static_issues": [], "findings": findings, "errors": []}


@needs_ruff
def test_static_node_only_changed_lines_and_window():
    old = "import sys\n" + "a = 1\n" * 30
    new = "import sys\nimport os\n" + "a = 1\n" * 30   # 'sys' unused too, but it is not a changed line
    out = sa.static_node(_state(old, new, [make_finding("r:f0", "a.py", 3), make_finding("r:f1", "a.py", 25)]))
    assert [(i["code"], i["row"], i["file"]) for i in out["static_issues"]] == [("F401", 2, "a.py")]
    assert [f.static_overlap for f in out["findings"]] == [1.0, None]
    assert out["errors"] == []


@needs_ruff
def test_static_node_undefined_name_on_changed_line():
    old = "def f():\n    return 1\n"
    new = "def f():\n    return missing_name\n"
    out = sa.static_node(_state(old, new, [make_finding("r:f0", "a.py", 2)]))
    assert out["static_issues"][0]["code"] == "F821"
    assert out["findings"][0].static_overlap == 1.0


def test_static_node_missing_source_is_unavailable_and_logged():
    out = sa.static_node(_state("a = 1\n", "a = 1\nb = 2\n", [make_finding("r:f0", "a.py", 2)], files={}))
    assert out["findings"][0].static_overlap is None
    assert out["static_issues"] == [] and "no source for a.py" in out["errors"][0]


def test_static_node_ruff_blocked_is_unavailable_and_logged(monkeypatch):
    def boom(*a, **k):
        raise PermissionError("blocked")
    monkeypatch.setattr(sa.subprocess, "run", boom)
    out = sa.static_node(_state("a = 1\n", "a = 1\nb = 2\n", [make_finding("r:f0", "a.py", 2)]))
    assert out["findings"][0].static_overlap is None and "static: a.py" in out["errors"][0]


def test_static_node_keeps_earlier_errors_and_no_diff_files():
    st = _state("a = 1\n", "a = 1\n", [make_finding()])
    st["errors"] = ["earlier"]
    out = sa.static_node(st)
    assert out["errors"] == ["earlier"] and out["findings"][0].static_overlap is None


# ---- benchmark replay

@needs_ruff
def test_replay_roundtrip(tmp_path):
    cases = tmp_path / "cases"
    (cases / "proj-1").mkdir(parents=True)
    old, new = "a = 1\n", "a = 1\nimport os\n"
    (cases / "proj-1" / "meta.json").write_text(json.dumps(
        {"project": "proj", "buggy_commit": "BUG", "fixed_commit": "FIX"}), encoding="utf-8")
    (cases / "proj-1" / "positive.diff").write_text(diff_adding("m.py", old, new), encoding="utf-8")
    seen = []

    def show(project, commit, path):
        seen.append((project, commit, path))
        return new

    rec = {"run_id": "proj-1:positive:s0", "case_id": "proj-1", "kind": "positive", "status": "ok",
           "findings": [make_finding("proj-1:positive:s0:f0", "m.py", 2).model_dump(mode="json"),
                        make_finding("proj-1:positive:s0:f1", "m.py", 30).model_dump(mode="json")]}
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(rec) + "\n" + json.dumps({"run_id": "x", "status": "error"}) + "\n", encoding="utf-8")
    out = tmp_path / "sig" / "static.jsonl"
    stats = sa.replay(runs, out, cases_dir=cases, show=show)
    assert seen == [("proj", "BUG", "m.py")]  # positive run reads the buggy commit
    assert (stats["runs"], stats["findings"], stats["overlap"]) == (1, 2, 1)
    line = json.loads(out.read_text(encoding="utf-8"))
    assert line["rules"] == list(config.STATIC_RULES) and line["ruff_version"].startswith("ruff")
    assert [f["static_overlap"] for f in line["findings"]] == [1.0, None]
    assert line["findings"][0]["matches"] == [{"code": "F401", "row": 2}]
    assert b"\r\n" not in out.read_bytes()


def test_replay_refuses_when_ruff_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "ruff_version", lambda: None)
    (tmp_path / "r.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(sa.StaticError, match="does not run"):
        sa.replay(tmp_path / "r.jsonl", tmp_path / "o.jsonl")
