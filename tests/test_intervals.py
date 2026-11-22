"""Wilson alongside the bootstrap, the always-escalate baseline, and the report's new sections."""
import json

import pytest

from benchmark import evaluate as ev, judge, labeling, metrics as mt
from benchmark.labeling import Labeled, Run

AUTO, ESC = "auto_post", "escalate"


def F(i, run_id, is_tp, decision=AUTO, score=0.9):
    case, kind, _ = run_id.split(":")
    return Labeled(id=f"f{i}", run_id=run_id, case_id=case, kind=kind, sample=0, file="m.py", line=1,
                   category="logic", severity="medium", score=score, decision=decision, location_match=True,
                   is_tp=is_tp, reason="t")


def R(run_id):
    case, kind, s = run_id.split(":")
    return Run(run_id, case, kind, 0)


def all_success(n=10):
    runs = [R(f"c{i}:{k}:s0") for i in range(n) for k in ("positive", "negative")]
    fs = [F(i, f"c{i}:positive:s0", True) for i in range(n)]
    return runs, fs


def test_degenerate_detects_zero_width_intervals():
    assert mt.degenerate((1.0, 1.0, 100)) and mt.degenerate((0.5, 0.5))
    assert not mt.degenerate((0.2, 0.4, 100)) and not mt.degenerate(None)


def test_summarize_adds_wilson_to_run_level_proportions():
    runs, fs = all_success(10)
    s = mt.summarize(runs, fs, B=100, seed=0)
    assert s["detection_recall"]["ci"] == (1.0, 1.0)                      # the collapsed bootstrap
    assert s["detection_recall"]["wilson"] == mt.wilson(10, 10)
    assert s["detection_recall"]["wilson"][0] == pytest.approx(0.7225, abs=1e-3)
    assert s["autopost_recall"]["wilson"] == mt.wilson(10, 10)
    assert s["fix_flag_rate"]["wilson"] == mt.wilson(0, 10)
    assert s["fp_load_auto"]["wilson"] is None                            # a rate, not a proportion


def test_curve_rows_carry_wilson_for_every_curve_metric():
    runs, fs = all_success(10)
    rows = mt.operating_curve(runs, fs, mt.score_at_least, [0.5, 0.95], B=30, seed=0)
    for r in rows:
        assert all((k + "_wilson") in r for k in mt.CURVE_KEYS)
    assert rows[0]["detection_recall_wilson"] == mt.wilson(10, 10)
    assert rows[1]["auto_post_precision_wilson"] is None                  # nothing auto-posted at 0.95: n = 0


def test_always_escalate_policy():
    runs, fs = all_success(10)
    m = mt.metrics(runs, mt.with_decisions(fs, mt.always_escalate))
    assert m["escalation_rate"] == 1.0 and m["auto_post_precision"] is None
    assert m["autopost_recall"] == 0.0 and m["detection_recall"] == 1.0   # the reviewer still found them
    assert mt.POLICIES["always_escalate"] is mt.always_escalate


# ---- through the CLI --------------------------------------------------------------------------
HTTPIE = labeling.load_cases(split="dev")["httpie-4"]["bug_locations"][0]


def write(tmp_path):
    f = HTTPIE["file"]
    fnd = lambda fid, line, score: dict(id=fid, file=f, line=line, category="logic", severity="medium", message="m",
                                       expected_failure=None, self_confidence=score)
    rec = lambda kind, fs: {"run_id": f"httpie-4:{kind}:s0", "case_id": "httpie-4", "kind": kind, "sample": 0,
                            "status": "ok", "error": None, "findings": fs, "est_tokens": 1}
    recs = [rec("positive", [fnd("httpie-4:positive:s0:f0", HTTPIE["start"], 0.95),
                             fnd("httpie-4:positive:s0:f1", HTTPIE["start"] + 300, 0.85)]),
            rec("negative", [fnd("httpie-4:negative:s0:f0", HTTPIE["start"], 0.85)])]
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"finding_id": "httpie-4:positive:s0:f0", "verdict": "yes", "status": "ok",
                                  "reason": "r", "prompt_version": judge.JUDGE_PROMPT_VERSION}), encoding="utf-8")
    return runs, labels


def test_report_flags_a_collapsed_bootstrap_and_prints_wilson(tmp_path, capsys):
    runs, labels = write(tmp_path)
    ev.main([str(runs), "--labels", str(labels), "--boot", "100"])
    line = next(l for l in capsys.readouterr().out.splitlines() if "detection recall" in l)
    assert "Wilson [0.21, 1.00]" in line and "bootstrap collapsed" in line      # 1 of 1 case


def test_always_escalate_block_and_calibration_section_in_the_report(tmp_path, capsys):
    runs, labels = write(tmp_path)
    ev.main([str(runs), "--labels", str(labels), "--boot", "100", "--policy", "always_escalate", "--curve",
             "--calibration"])
    out = capsys.readouterr().out
    assert "ALWAYS ESCALATE (a human reviews everything)" in out and "auto-posted 0, escalated 3" in out
    assert "w = bootstrap collapsed, Wilson shown" in out
    assert "CALIBRATION of self_confidence (all 3 scored findings" in out
    assert "Brier score" in out and "ECE (<=5 equal-mass bins)" in out and "AUROC (TP above FP)" in out
    assert "reliability:" in out