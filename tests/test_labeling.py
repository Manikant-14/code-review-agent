"""Tests for benchmark/labeling.py: matcher, dataset building, judge queue, real-case regressions."""
import pytest

from benchmark import labeling as lb

LOC = [{"file": "pkg/m.py", "start": 10, "end": 12, "anchor_only": False}]
CASES = {"c1": {"project": "p", "split": "dev", "bug_locations": LOC}}


def finding(fid, line, file="pkg/m.py", **kw):
    d = dict(id=fid, file=file, line=line, category="logic", severity="medium",
             message="x", expected_failure=None, self_confidence=0.8)
    d.update(kw)
    return d


def rec(run_id, kind, findings, case="c1", status="ok", sample=0):
    return {"run_id": run_id, "case_id": case, "kind": kind, "sample": sample, "status": status,
            "error": None if status == "ok" else "boom", "findings": findings}


def records():
    rs = [
        rec("c1:positive:s0", "positive", [finding("a", 11), finding("b", 40), finding("c", 11, file="other.py")]),
        rec("c1:negative:s0", "negative", [finding("d", 11)]),
        rec("c2:positive:s0", "positive", [finding("e", 11)], case="c2"),
        rec("c1:positive:s1", "positive", [], status="error", sample=1),
    ]
    return {r["run_id"]: r for r in rs}


# ---- matcher ----------------------------------------------------------------------------------
def test_window_edges_are_inclusive():
    assert lb.location_match("pkg/m.py", 7, LOC, 3)      # start - W
    assert not lb.location_match("pkg/m.py", 6, LOC, 3)
    assert lb.location_match("pkg/m.py", 15, LOC, 3)     # end + W
    assert not lb.location_match("pkg/m.py", 16, LOC, 3)


def test_default_window_comes_from_config():
    assert lb.location_match("pkg/m.py", 7, LOC) and not lb.location_match("pkg/m.py", 6, LOC)


def test_anchor_only_location_uses_start_equals_end():
    anchor = [{"file": "a.py", "start": 20, "end": 20, "anchor_only": True}]
    assert lb.location_match("a.py", 23, anchor) and not lb.location_match("a.py", 24, anchor)


def test_any_location_of_the_bug_counts():
    locs = LOC + [{"file": "pkg/n.py", "start": 100, "end": 100, "anchor_only": True}]
    assert lb.location_match("pkg/n.py", 101, locs)


@pytest.mark.parametrize("finding_file, ok", [
    ("pkg/m.py", True), ("models.py", False), ("m.py", True), (".\\pkg\\m.py", True),
    ("./pkg/m.py", True), ("src/pkg/m.py", True), ("g/m.py", False), ("pkg/other.py", False)])
def test_same_file_matches_on_path_boundaries(finding_file, ok):
    assert lb.same_file(finding_file, "pkg/m.py") is ok


def test_suffix_without_boundary_does_not_match():
    assert not lb.same_file("s.py", "httpie/models.py")      # bakeoff.is_hit would say yes


# ---- dataset ----------------------------------------------------------------------------------
def by_id(fs):
    return {f.id: f for f in fs}


def test_labels_without_verdicts_leave_location_matches_pending():
    runs, fs, skipped = lb.build_dataset(records(), CASES)
    f = by_id(fs)
    assert [r.run_id for r in runs] == ["c1:negative:s0", "c1:positive:s0"]
    assert (f["a"].is_tp, f["a"].reason) == (None, "pending_judge") and f["a"].location_match
    assert (f["b"].is_tp, f["b"].reason) == (False, "location") and not f["b"].location_match
    assert (f["c"].is_tp, f["c"].reason) == (False, "location")
    assert (f["d"].is_tp, f["d"].reason) == (False, "negative_run") and f["d"].location_match


def test_judge_verdicts_decide_location_matches():
    _, yes, _ = lb.build_dataset(records(), CASES, {"a": "yes"})
    _, no, _ = lb.build_dataset(records(), CASES, {"a": "no"})
    assert (by_id(yes)["a"].is_tp, by_id(yes)["a"].reason) == (True, "judge_yes")
    assert (by_id(no)["a"].is_tp, by_id(no)["a"].reason) == (False, "judge_no")


def test_a_verdict_cannot_rescue_a_finding_that_missed_the_location():
    _, fs, _ = lb.build_dataset(records(), CASES, {"b": "yes", "c": "yes", "d": "yes"})
    assert not any(by_id(fs)[k].is_tp for k in "bcd")


def test_other_split_and_failed_runs_are_skipped_and_listed():
    runs, fs, skipped = lb.build_dataset(records(), CASES)
    assert dict(skipped)["c2:positive:s0"] == "case not in this split"
    assert dict(skipped)["c1:positive:s1"].startswith("run failed")
    assert "e" not in by_id(fs)


def test_score_and_decision_are_carried():
    r = records()
    r["c1:positive:s0"]["findings"][0].update(confidence=0.42, decision="auto_post")
    _, fs, _ = lb.build_dataset(r, CASES, score_key="confidence")
    assert by_id(fs)["a"].score == 0.42 and by_id(fs)["a"].decision == "auto_post"
    assert by_id(fs)["b"].score is None and by_id(fs)["b"].decision is None


def test_bad_decision_and_bad_verdict_are_rejected():
    r = records()
    r["c1:positive:s0"]["findings"][0]["decision"] = "maybe"
    with pytest.raises(ValueError):
        lb.build_dataset(r, CASES)
    with pytest.raises(ValueError):
        lb.build_dataset(records(), CASES, {"a": "maybe"})


def test_judge_queue_is_location_matched_positive_findings_without_a_verdict():
    q = lb.judge_queue(records(), CASES)
    assert [x["finding_id"] for x in q] == ["a"] and q[0]["case_id"] == "c1"
    assert lb.judge_queue(records(), CASES, {"a": "no"}) == []


def test_load_latest_records_keeps_retry_and_skips_half_lines(tmp_path):
    import json
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(rec("x", "positive", [], status="error")) + "\n"
                 + json.dumps(rec("x", "positive", [])) + "\n" + '{"run_id": "half', encoding="utf-8")
    assert lb.load_latest_records(p)["x"]["status"] == "ok"


# ---- real ground truth ------------------------------------------------------------------------
def test_real_cases_load_and_split_filter():
    dev = lb.load_cases(split="dev")
    assert len(dev) >= 51 and "httpie-4" in dev
    assert lb.load_cases(split="test") == {}


def test_real_ground_truth_regressions():
    dev = lb.load_cases(split="dev")
    assert lb.location_match("httpie/models.py", 106, dev["httpie-4"]["bug_locations"])
    # thefuck-10: the missing "No manual entry" check is an anchor added in step 5b
    assert lb.location_match("thefuck/rules/man.py", 14, dev["thefuck-10"]["bug_locations"])
    assert not lb.location_match("thefuck/rules/man.py", 60, dev["thefuck-10"]["bug_locations"])