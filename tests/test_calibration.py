"""Tests for benchmark/calibration.py: hand-computed Brier, ECE, AUROC, ties, bootstrap."""
import random

import pytest

from benchmark import calibration as cal
from benchmark.labeling import Labeled, Run


def F(i, score, is_tp, case="c1", kind="positive", run=None):
    return Labeled(id=f"f{i}", run_id=run or f"{case}:{kind}:s0", case_id=case, kind=kind, sample=0, file="m.py",
                   line=1, category="logic", severity="medium", score=score, decision=None, location_match=True,
                   is_tp=is_tp, reason="t")


def test_brier_reference_and_skill():
    assert cal.brier([(1.0, 1), (0.0, 0)]) == 0.0
    assert cal.brier_skill([(1.0, 1), (0.0, 0)]) == 1.0
    assert cal.brier([(0.5, 1), (0.5, 0)]) == 0.25 and cal.brier_reference([(0.5, 1), (0.5, 0)]) == 0.25
    assert cal.brier_skill([(0.5, 1), (0.5, 0)]) == 0.0
    assert cal.brier_skill([(0.9, 1), (0.9, 1)]) is None          # base-rate predictor is perfect: skill undefined
    assert cal.brier([]) is None and cal.brier_reference([]) is None


def test_overconfident_scores_have_negative_skill():
    pairs = [(0.9, 1), (0.9, 0), (0.9, 0), (0.9, 0)]               # says 90% but is right 25% of the time
    assert cal.brier_skill(pairs) < 0


def test_reliability_and_ece_hand_computed():
    pairs = [(0.2, 0), (0.2, 0), (0.8, 1), (0.8, 0)]
    bins = cal.reliability(pairs, k=2)
    assert [b["n"] for b in bins] == [2, 2]
    assert (bins[0]["mean_score"], bins[0]["rate"]) == (0.2, 0.0)
    assert (bins[1]["mean_score"], bins[1]["rate"]) == (0.8, 0.5)
    assert cal.ece(pairs, k=2) == pytest.approx(0.5 * 0.2 + 0.5 * 0.3)


def test_identical_scores_never_straddle_bins():
    pairs = [(0.85, i % 2) for i in range(10)]
    bins = cal.reliability(pairs, k=5)
    assert len(bins) == 1 and bins[0]["n"] == 10
    assert cal.ece(pairs, 5) == pytest.approx(abs(0.5 - 0.85))


def test_bins_never_share_a_score_and_cover_everything():
    rng = random.Random(3)
    pairs = [(rng.choice([0.6, 0.7, 0.8, 0.85, 0.9, 0.95]), rng.randint(0, 1)) for _ in range(60)]
    bins = cal.reliability(pairs, k=5)
    assert sum(b["n"] for b in bins) == 60 and len(bins) <= 5
    for a, b in zip(bins, bins[1:]):
        assert a["hi"] < b["lo"]


def test_auroc_hand_computed_and_edge_cases():
    assert cal.auroc([(0.9, 1), (0.5, 1), (0.7, 0), (0.1, 0)]) == 0.75
    assert cal.auroc([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]) == 1.0
    assert cal.auroc([(0.1, 1), (0.2, 1), (0.8, 0), (0.9, 0)]) == 0.0
    assert cal.auroc([(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]) == 0.5       # all tied
    assert cal.auroc([(0.9, 1), (0.8, 1)]) is None and cal.auroc([(0.9, 0)]) is None and cal.auroc([]) is None


def test_rank_based_auroc_matches_brute_force_with_ties():
    rng = random.Random(11)
    pairs = [(round(rng.random(), 1), rng.randint(0, 1)) for _ in range(80)]
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    brute = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    assert cal.auroc(pairs) == pytest.approx(brute)


def test_a_calibrated_score_scores_well():
    rng = random.Random(5)
    pairs = []
    for _ in range(3000):
        p = rng.choice([0.2, 0.4, 0.6, 0.8])
        pairs.append((p, 1 if rng.random() < p else 0))
    assert cal.ece(pairs, 5) < 0.04 and cal.brier_skill(pairs) > 0 and cal.auroc(pairs) > 0.6


def test_findings_without_a_score_are_ignored_and_pending_labels_raise():
    fs = [F(1, 0.9, True), F(2, None, False), F(3, 0.5, False)]
    m = cal.calibration_metrics(fs)
    assert m["n"] == 2 and m["base_rate"] == 0.5
    with pytest.raises(ValueError, match="judge"):
        cal.calibration_metrics([F(1, 0.9, None)])
    assert cal.calibration_metrics([])["brier"] is None


def dataset():
    runs, fs = [], []
    rng = random.Random(2)
    for i in range(20):
        case = f"c{i}"
        runs += [Run(f"{case}:positive:s0", case, "positive", 0), Run(f"{case}:negative:s0", case, "negative", 0)]
        tp = rng.random() < 0.7
        fs.append(F(len(fs), rng.choice([0.85, 0.9, 0.95]), tp, case=case))
        if rng.random() < 0.4:
            fs.append(F(len(fs), 0.85, False, case=case, kind="negative"))
    return runs, fs


def test_summary_is_deterministic_and_has_ordered_intervals():
    runs, fs = dataset()
    a = cal.calibration_summary(runs, fs, B=150, seed=1)
    b = cal.calibration_summary(runs, fs, B=150, seed=1)
    assert a == b
    for key in cal.KEYS:
        lo, hi, n = a["ci"][key]
        assert lo <= hi and 0 < n <= 150
    assert a["ci"]["auroc"][0] >= 0.0 and a["ci"]["auroc"][1] <= 1.0