"""Tests for benchmark/judge.py (the LLM is faked: no network, no keys)."""
import json
import threading

import pytest
from pydantic import ValidationError

from benchmark import judge as jd
from src import config, llm


def item(fid="c1:positive:s0:f0", case="c1", **kw):
    f = dict(id=fid, file="pkg/m.py", line=11, message="the check is missing", expected_failure="KeyError",
             self_confidence=0.91234, severity="sevmark", category="catmark", decision="auto_post")
    f.update(kw)
    return {"finding_id": fid, "case_id": case, "finding": f}


def write_case(tmp_path, case="c1", patch="--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1 +1 @@\n-old\n+new\n"):
    d = tmp_path / case
    d.mkdir()
    (d / "negative.diff").write_text(patch, encoding="utf-8")
    return tmp_path


# ---- verdict and prompt -----------------------------------------------------------------------
def test_verdict_is_normalised_and_validated():
    assert jd.Verdict(verdict=" Yes ", reason="r").verdict == "yes"
    assert jd.Verdict(verdict="NO", reason="r").verdict == "no"
    with pytest.raises(ValidationError):
        jd.Verdict(verdict="maybe", reason="r")


def test_prompt_has_patch_and_finding_text_but_nothing_else():
    p = jd.build_prompt("PATCHTEXT\n", item()["finding"])
    assert "PATCHTEXT" in p and "pkg/m.py" in p and "line: 11" in p
    assert "the check is missing" in p and "KeyError" in p
    for secret in ("0.91234", "sevmark", "catmark", "auto_post", "confidence"):
        assert secret not in p and secret not in jd.SYSTEM


def test_prompt_says_none_given_when_there_is_no_expected_failure():
    assert "expected failure: none given" in jd.build_prompt("p", item(expected_failure=None)["finding"])


# ---- judge_finding ----------------------------------------------------------------------------
def test_judge_finding_uses_the_judge_role_and_the_stored_patch(tmp_path, monkeypatch):
    seen = {}

    def fake(role, prompt, *, system=None, schema=None, sample=0, use_cache=True):
        seen.update(role=role, prompt=prompt, system=system, schema=schema)
        return jd.Verdict(verdict="yes", reason=" same defect ")

    monkeypatch.setattr(llm, "call_llm", fake)
    rec = jd.judge_finding(item(), cases_dir=write_case(tmp_path))
    assert seen["role"] == "judge" and seen["system"] == jd.SYSTEM and seen["schema"] is jd.Verdict
    assert "+new" in seen["prompt"]
    assert rec["verdict"] == "yes" and rec["reason"] == "same defect" and rec["status"] == "ok"
    assert rec["prompt_version"] == jd.JUDGE_PROMPT_VERSION and rec["judge_model"] == config.MODELS["judge"]
    json.dumps(rec)


# ---- verdict file -----------------------------------------------------------------------------
def rec(fid, verdict="yes", status="ok", version=None):
    return {"finding_id": fid, "case_id": "c1", "verdict": verdict, "reason": "r", "status": status,
            "prompt_version": version or jd.JUDGE_PROMPT_VERSION, "error": None}


def test_load_verdicts_keeps_current_version_ok_records_only(tmp_path):
    p = tmp_path / "labels.jsonl"
    lines = [rec("a", "yes"), rec("b", "no"), rec("c", None, status="error"), rec("d", "yes", version="old"),
             rec("a", "no")]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n" + '{"finding_id": "half', encoding="utf-8")
    assert jd.load_verdicts(p) == {"a": "no", "b": "no"}          # the last ok record wins
    assert jd.load_verdicts(tmp_path / "missing.jsonl") == {}


# ---- run_judge --------------------------------------------------------------------------------
def fake_judge(it, **_):
    return rec(it["finding_id"], "yes")


def test_run_judge_writes_one_line_per_finding_and_resumes(tmp_path):
    out = tmp_path / "labels.jsonl"
    queue = [item(f"c1:positive:s0:f{i}") for i in range(6)]
    s1 = jd.run_judge(queue, out, workers=4, judge_fn=fake_judge)
    assert (s1["ok"], s1["error"], s1["skipped"]) == (6, 0, 0)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(x["finding_id"] for x in lines) == sorted(q["finding_id"] for q in queue)
    s2 = jd.run_judge(queue, out, workers=4, judge_fn=fake_judge)
    assert (s2["ok"], s2["skipped"]) == (0, 6) and len(out.read_text(encoding="utf-8").splitlines()) == 6


def test_run_judge_really_runs_in_parallel(tmp_path):
    barrier = threading.Barrier(4, timeout=5)

    def waits_for_peers(it, **_):
        barrier.wait()                                    # only passes if 4 calls are in flight together
        return rec(it["finding_id"])

    queue = [item(f"c1:positive:s0:f{i}") for i in range(4)]
    assert jd.run_judge(queue, tmp_path / "l.jsonl", workers=4, judge_fn=waits_for_peers)["ok"] == 4


def test_errors_are_recorded_and_retried_next_run(tmp_path):
    out = tmp_path / "labels.jsonl"
    calls = []

    def flaky(it, **_):
        calls.append(it["finding_id"])
        if it["finding_id"].endswith("f1") and calls.count(it["finding_id"]) == 1:
            raise RuntimeError("Google 500")
        return rec(it["finding_id"])

    queue = [item(f"c1:positive:s0:f{i}") for i in range(3)]
    s1 = jd.run_judge(queue, out, workers=2, judge_fn=flaky)
    assert (s1["ok"], s1["error"]) == (2, 1)
    assert "Google 500" in [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()
                            if '"error"' in x and "Google" in x][0]["error"]
    s2 = jd.run_judge(queue, out, workers=2, judge_fn=flaky)
    assert (s2["ok"], s2["error"], s2["skipped"]) == (1, 0, 2)
    assert set(jd.load_verdicts(out)) == {q["finding_id"] for q in queue}


def test_daily_quota_stops_new_work_and_writes_nothing_for_it(tmp_path):
    out = tmp_path / "labels.jsonl"

    def quota(it, **_):
        raise llm.DailyQuotaExhausted("RPD")

    s = jd.run_judge([item(f"c1:positive:s0:f{i}") for i in range(3)], out, workers=1, judge_fn=quota)
    assert s["ok"] == 0 and "daily quota" in s["stopped"]
    assert not out.exists() or out.read_text(encoding="utf-8") == ""


def test_verdicts_of_another_prompt_version_are_ignored(tmp_path):
    out = tmp_path / "labels.jsonl"
    jd.run_judge([item()], out, judge_fn=fake_judge)
    assert jd.load_verdicts(out, jd.JUDGE_PROMPT_VERSION) == {"c1:positive:s0:f0": "yes"}
    assert jd.load_verdicts(out, "some-other-version") == {}


def test_prompt_tells_the_judge_the_reviewer_saw_the_change_reversed():
    assert "opposite direction" in jd.SYSTEM and "direction of the wording" in jd.SYSTEM
    assert jd.JUDGE_PROMPT_VERSION != "j1"          # j1 lacked this note; its verdicts must not be reused
