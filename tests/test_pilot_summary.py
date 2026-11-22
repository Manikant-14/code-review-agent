"""Tests for benchmark/pilot_summary.py (synthetic runs and cases only)."""
import json

from benchmark import pilot_summary as ps


def finding(line, conf=0.5, file="m.py"):
    return {"file": file, "line": line, "self_confidence": conf}


def rec(cid, kind, findings, status="ok", est=100):
    return {"run_id": f"{cid}:{kind}:s0", "case_id": cid, "kind": kind, "status": status,
            "findings": findings, "est_tokens": est, "error": None if status == "ok" else "boom"}


def test_summarise_counts_hits_fixflags_and_confidences(tmp_path):
    d = tmp_path / "c1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"bug_locations": [{"file": "m.py", "start": 10, "end": 10}]}))
    latest = {r["run_id"]: r for r in [
        rec("c1", "positive", [finding(11, 0.9), finding(40, 0.2)]),
        rec("c1", "negative", [finding(10, 0.8)]),
        rec("c2", "negative", [], status="error"),
    ]}
    s = ps.summarise(latest, cases_dir=tmp_path)
    assert s["cases"]["c1"]["pos"] == (True, 2) and s["cases"]["c1"]["neg"] == 1
    assert s["hit_conf"] == [0.9] and sorted(s["other_conf"]) == [0.2, 0.8]
    assert s["errors"] == [("c2:negative:s0", "boom")]


def test_far_finding_is_not_a_hit(tmp_path):
    d = tmp_path / "c1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"bug_locations": [{"file": "m.py", "start": 10, "end": 10}]}))
    s = ps.summarise({"c1:positive:s0": rec("c1", "positive", [finding(30)])}, cases_dir=tmp_path)
    assert s["cases"]["c1"]["pos"] == (False, 1)


def test_load_latest_keeps_retry_over_error(tmp_path):
    p = tmp_path / "r.jsonl"
    lines = [rec("c1", "positive", [], status="error"), rec("c1", "positive", [])]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n" + "{half", encoding="utf-8")
    assert ps.load_latest(p)["c1:positive:s0"]["status"] == "ok"