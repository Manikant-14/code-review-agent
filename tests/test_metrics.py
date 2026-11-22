"""Tests for benchmark/metrics.py: hand-computed metrics, intervals, policies, bootstrap, curve."""
import pytest

from benchmark import metrics as mt
from benchmark.labeling import Labeled, Run

AUTO, ESC = "auto_post", "escalate"


def F(fid, run_id, is_tp, decision=None, score=0.5, sev="medium", cat="logic"):
    case, kind, _ = run_id.split(":")
    return Labeled(id=fid, run_id=run_id, case_id=case, kind=kind, sample=0, file="m.py", line=1,
                   category=cat, severity=sev, score=score, decision=decision, location_match=True,
                   is_tp=is_tp, reason="t")


def R(run_id):
    case, kind, s = run_id.split(":")
    return Run(run_id, case, kind, int(s[1:]))


RUNS = [R(x) for x in ("c1:positive:s0", "c1:negative:s0", "c2:positive:s0", "c2:negative:s0", "c3:positive:s0")]
FINDINGS = [
    F("f1", "c1:positive:s0", True, AUTO),
    F("f2", "c1:positive:s0", False, ESC),
    F("f3", "c1:negative:s0", False, AUTO),
    F("f4", "c3:positive:s0", True, ESC),
    F("f5", "c3:positive:s0", False, AUTO),      # duplicate-ish finding: counts for precision
]


# ---- intervals --------------------------------------------------------------------------------
def test_wilson_known_values():
    assert mt.wilson(0, 0) is None
    lo, hi = mt.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = mt.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = mt.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3) and hi == 1.0


def test_percentile():
    assert mt.percentile([], 0.5) is None
    assert mt.percentile([1, 2, 3, 4], 0.0) == 1 and mt.percentile([1, 2, 3, 4], 1.0) == 4
    assert mt.percentile([1, 2, 3, 4], 0.5) == 2.5


# ---- metrics ----------------------------------------------------------------------------------
def test_metrics_hand_computed():
    m = mt.metrics(RUNS, FINDINGS)
    assert (m["n_pos_runs"], m["n_neg_runs"], m["n_findings"]) == (3, 2, 5)
    assert (m["n_auto"], m["n_escalated"]) == (3, 2)
    assert m["auto_post_precision"] == pytest.approx(1 / 3)        # f1 of f1, f3, f5
    assert m["escalation_rate"] == pytest.approx(2 / 5)
    assert m["escalation_rate_pos"] == pytest.approx(2 / 4)
    assert m["escalation_rate_neg"] == 0.0
    assert m["detection_recall"] == pytest.approx(2 / 3)           # c1 and c3 have a true positive
    assert m["autopost_recall"] == pytest.approx(1 / 3)            # only c1's is auto-posted
    assert m["fp_load_auto"] == pytest.approx(1 / 2)               # f3 over two negative runs
    assert m["fp_load_escalated"] == 0.0
    assert m["fix_flag_rate"] == pytest.approx(1 / 2)              # c1's negative run had a finding


def test_runs_without_findings_count_in_run_level_denominators():
    m = mt.metrics(RUNS, FINDINGS)
    assert m["n_pos_runs"] == 3                                    # c2:positive has no findings


def test_undefined_metrics_are_none_not_zero():
    m = mt.metrics([], [])
    assert all(m[k] is None for k in ("auto_post_precision", "escalation_rate", "detection_recall",
                                      "autopost_recall", "fp_load_auto", "fix_flag_rate"))
    only_esc = mt.metrics(RUNS, [F("f", "c1:positive:s0", True, ESC)])
    assert only_esc["auto_post_precision"] is None and only_esc["escalation_rate"] == 1.0


def test_pending_label_and_missing_decision_raise():
    with pytest.raises(ValueError, match="judge"):
        mt.metrics(RUNS, [F("f", "c1:positive:s0", None, AUTO)])
    with pytest.raises(ValueError, match="decision"):
        mt.metrics(RUNS, [F("f", "c1:positive:s0", True, None)])


# ---- policies ---------------------------------------------------------------------------------
def test_all_auto_has_zero_escalation():
    m = mt.metrics(RUNS, mt.with_decisions(FINDINGS, mt.all_auto))
    assert m["escalation_rate"] == 0.0 and m["n_auto"] == 5


def test_score_threshold_policy_and_missing_score():
    fs = [F("a", "c1:positive:s0", True, score=0.8), F("b", "c1:positive:s0", False, score=0.8 - 0.01),
          F("c", "c1:positive:s0", False, score=None)]
    out = mt.with_decisions(fs, mt.score_at_least, 0.8)
    assert [f.decision for f in out] == [AUTO, ESC, ESC]


def test_severity_scaled_policy():
    f = lambda **kw: F("x", "c1:positive:s0", True, **kw)
    assert mt.severity_scaled(f(score=0.8), 1.0) and not mt.severity_scaled(f(score=0.7), 1.0)   # medium 0.75
    assert mt.severity_scaled(f(score=0.7), 0.5)                                                   # 0.375
    assert not mt.severity_scaled(f(score=0.85, sev="low", cat="security"), 1.0)                   # floor: HIGH 0.90
    assert mt.severity_scaled(f(score=0.65, sev="low"), 1.0)                                       # low 0.65
    assert mt.severity_scaled(f(score=1.0, sev="high"), 2.0)                                       # capped at 1.0
    assert not mt.severity_scaled(f(score=None), 1.0)


# ---- bootstrap --------------------------------------------------------------------------------
def test_bootstrap_resamples_whole_cases_and_gives_copies_their_own_run_ids():
    runs = [R(f"c{i}:{k}:s0") for i in range(5) for k in ("positive", "negative")]
    fs = [F(f"a{j}", "c0:positive:s0", True, AUTO) for j in range(10)]   # only case c0 has findings
    seen = set()
    for rs, fb in mt.bootstrap_samples(runs, fs, B=200, seed=1):
        assert len(fb) % 10 == 0                   # a finding is never drawn without its whole case
        assert len({r.run_id for r in rs}) == len(rs) == 10
        seen.add(len(fb))
    assert 0 in seen and max(seen) > 10


def test_bootstrap_is_deterministic_for_a_seed():
    a = mt.cluster_bootstrap(RUNS, FINDINGS, B=100, seed=3)
    b = mt.cluster_bootstrap(RUNS, FINDINGS, B=100, seed=3)
    c = mt.cluster_bootstrap(RUNS, FINDINGS, B=100, seed=4)
    assert a == b and a != c


def test_bootstrap_interval_brackets_the_estimate():
    runs, fs = [], []
    for i in range(20):
        rid = f"c{i}:positive:s0"
        runs.append(R(rid))
        if i < 15:
            fs.append(F(f"f{i}", rid, True, AUTO))
    ci = mt.cluster_bootstrap(runs, fs, ("detection_recall",), B=500, seed=0)["detection_recall"]
    assert ci[0] <= 0.75 <= ci[1] and ci[2] == 500


def test_bootstrap_metric_never_defined_is_none():
    runs = [R("c1:positive:s0")]
    assert mt.cluster_bootstrap(runs, [], ("fp_load_auto",), B=20)["fp_load_auto"] is None


def test_summarize_uses_wilson_for_finding_proportions_and_bootstrap_for_run_level():
    s = mt.summarize(RUNS, FINDINGS, B=100, seed=0)
    assert s["auto_post_precision"]["method"] == "wilson" and s["auto_post_precision"]["ci"] == mt.wilson(1, 3)
    assert s["detection_recall"]["method"] == "bootstrap" and s["detection_recall"]["n_valid"] == 100
    assert s["counts"]["n_findings"] == 5


# ---- operating curve --------------------------------------------------------------------------
def curve_data():
    runs, fs = [], []
    for i in range(10):
        rid = f"c{i}:positive:s0"
        runs.append(R(rid))
        fs += [F(f"t{i}", rid, True, score=0.9), F(f"p{i}", rid, False, score=0.5)]
    return runs, fs


def test_operating_curve_shape_and_values():
    runs, fs = curve_data()
    rows = mt.operating_curve(runs, fs, mt.score_at_least, [0.0, 0.3, 0.7, 0.95], B=50, seed=0)
    by = {r["param"]: r for r in rows}
    assert by[0.0]["escalation_rate"] == 0.0 and by[0.0]["auto_post_precision"] == 0.5
    assert by[0.7]["auto_post_precision"] == 1.0 and by[0.7]["escalation_rate"] == 0.5
    assert by[0.95]["auto_post_precision"] is None and by[0.95]["escalation_rate"] == 1.0
    assert by[0.95]["auto_post_precision_ci"] is None
    esc = [r["escalation_rate"] for r in rows]
    assert esc == sorted(esc)                       # raising the threshold never reduces escalation
    lo, hi, n = by[0.7]["auto_post_precision_ci"]
    assert lo <= 1.0 <= hi and n == 50


def test_operating_curve_ignores_existing_decisions():
    runs, fs = curve_data()
    fs = [Labeled(**{**f.__dict__, "decision": ESC}) for f in fs]
    rows = mt.operating_curve(runs, fs, mt.score_at_least, [0.0], B=5)
    assert rows[0]["escalation_rate"] == 0.0


def test_spread():
    assert mt.spread([None, None]) is None
    assert mt.spread([0.5, None, 0.7, 0.9]) == {"min": 0.5, "mean": pytest.approx(0.7), "max": 0.9, "n": 3}