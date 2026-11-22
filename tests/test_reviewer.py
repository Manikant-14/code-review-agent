"""Tests for src/reviewer.py (the LLM is faked: no network, no keys)."""
import json

import pytest
from pydantic import ValidationError

from src import config, llm, reviewer as rv
from src.state import Finding


def item(**kw):
    d = dict(file="a/b.py", line=7, category="logic", severity="medium", message=" boom ",
             expected_failure="KeyError", self_confidence=0.7)
    d.update(kw)
    return d


def test_enum_and_confidence_are_normalised():
    f = rv.ReviewerFinding(**item(category=" Error-Handling ", severity="HIGH", self_confidence=85))
    assert f.category == config.Category.ERROR_HANDLING and f.severity == config.Severity.HIGH
    assert f.self_confidence == 0.85
    assert rv.ReviewerFinding(**item(self_confidence=-3)).self_confidence == 0.0
    assert rv.ReviewerFinding(**item(self_confidence=0.0)).self_confidence == 0.0


@pytest.mark.parametrize("raw", ["", "null", "None", "n/a", None])
def test_empty_expected_failure_becomes_none(raw):
    assert rv.ReviewerFinding(**item(expected_failure=raw)).expected_failure is None


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        rv.ReviewerFinding(**item(category="vibes"))


def test_to_findings_ids_paths_and_cap():
    items = [rv.ReviewerFinding(**item(file=".\\pkg\\m.py", line=i)) for i in range(5)]
    out = rv.to_findings("c-1:positive:s0", items)
    assert len(out) == rv.MAX_FINDINGS == 3
    assert [f.id for f in out] == ["c-1:positive:s0:f0", "c-1:positive:s0:f1", "c-1:positive:s0:f2"]
    assert out[0].file == "pkg/m.py" and out[0].message == "boom"
    assert isinstance(out[0], Finding) and out[0].decision is None and out[0].sandbox_verified is None
    assert out[0].self_confidence == 0.7


def test_prompt_names_every_category_and_severity():
    for c in config.Category:
        assert c.value in rv.SYSTEM
    for s in config.Severity:
        assert s.value in rv.SYSTEM


def fake_record(cid, kind, s, ctx=True, n=1):
    return {"run_id": rv.make_run_id(cid, kind, s), "case_id": cid, "kind": kind, "sample": s,
            "status": "ok", "findings": [{}] * n, "error": None}


def test_review_run_calls_reviewer_role_and_builds_record(tmp_path, monkeypatch):
    d = tmp_path / "p-1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"project": "p", "buggy_commit": "b", "fixed_commit": "f"}))
    diff = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
    (d / "positive.diff").write_text(diff)
    seen = {}

    def fake_call(role, prompt, *, system=None, schema=None, sample=0, use_cache=True):
        seen.update(role=role, system=system, schema=schema, sample=sample, prompt=prompt)
        return rv.Review(findings=[rv.ReviewerFinding(**item())])

    monkeypatch.setattr(llm, "call_llm", fake_call)
    rec = rv.review_run("p-1", "positive", sample=2, cases_dir=tmp_path, read_file=lambda p: "x = 2\n")
    assert seen["role"] == "reviewer" and seen["system"] == rv.SYSTEM and seen["schema"] is rv.Review
    assert seen["sample"] == 2 and "=== context" in seen["prompt"]
    assert rec["run_id"] == "p-1:positive:s2" and rec["status"] == "ok" and rec["with_context"]
    assert rec["prompt_version"] == rv.PROMPT_VERSION and rec["model"] == config.MODELS["reviewer"]
    assert rec["findings"][0]["id"] == "p-1:positive:s2:f0"
    json.dumps(rec)  # must be JSON-serialisable as is


def test_run_reviewer_writes_resumes_and_retries_errors(tmp_path):
    out = tmp_path / "runs.jsonl"
    calls = []

    def flaky(cid, kind, s, ctx):
        calls.append((cid, kind, s))
        if (cid, kind) == ("c2", "negative") and calls.count((cid, kind, s)) == 1:
            raise RuntimeError("Groq 500")
        return fake_record(cid, kind, s)

    s1 = rv.run_reviewer(["c1", "c2"], out_path=out, review_fn=flaky)
    assert (s1["ok"], s1["error"], s1["skipped"]) == (3, 1, 0)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4 and [x["status"] for x in lines].count("error") == 1
    assert "Groq 500" in [x for x in lines if x["status"] == "error"][0]["error"]

    s2 = rv.run_reviewer(["c1", "c2"], out_path=out, review_fn=flaky)  # resume
    assert (s2["ok"], s2["error"], s2["skipped"]) == (1, 0, 3)
    assert calls.count(("c2", "negative", 0)) == 2 and calls.count(("c1", "positive", 0)) == 1


def test_samples_expand_runs(tmp_path):
    s = rv.run_reviewer(["c1"], kinds=("positive",), samples=3, out_path=tmp_path / "r.jsonl",
                        review_fn=lambda cid, kind, smp, ctx: fake_record(cid, kind, smp))
    assert s["ok"] == 3


def test_daily_quota_stops_cleanly_and_records_nothing_for_that_run(tmp_path):
    out = tmp_path / "r.jsonl"

    def quota(cid, kind, s, ctx):
        if cid == "c2":
            raise llm.DailyQuotaExhausted("TPD")
        return fake_record(cid, kind, s)

    s = rv.run_reviewer(["c1", "c2", "c3"], out_path=out, review_fn=quota)
    assert s["ok"] == 2 and "c2:positive:s0" in s["stopped"]
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_half_written_last_line_is_ignored(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps(fake_record("c1", "positive", 0)) + "\n" + '{"run_id": "c1:neg', encoding="utf-8")
    assert rv._done_run_ids(out) == {"c1:positive:s0"}