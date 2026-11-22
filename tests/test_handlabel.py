"""Tests for benchmark/handlabel.py: blinded sampling, files, kappa and agreement."""
import csv
import json

import pytest

from benchmark import handlabel as ho, judge, labeling


def finding(fid, line=10, **kw):
    d = dict(id=fid, file="pkg/m.py", line=line, category="logic", severity="medium",
             message=f"msg {fid}", expected_failure=None, self_confidence=0.9)
    d.update(kw)
    return d


def many_records(n=20):
    pos = {"run_id": "c1:positive:s0", "case_id": "c1", "kind": "positive", "sample": 0, "status": "ok",
           "findings": [finding(f"c1:positive:s0:f{i}") for i in range(n)]}
    neg = {"run_id": "c1:negative:s0", "case_id": "c1", "kind": "negative", "sample": 0, "status": "ok",
           "findings": [finding(f"c1:negative:s0:f{i}") for i in range(4)]}
    return {pos["run_id"]: pos, neg["run_id"]: neg}


CASES = {"c1": {"project": "p", "split": "dev", "bug_locations": []}}


def verdicts(n=20):
    return {f"c1:positive:s0:f{i}": ("yes" if i % 2 == 0 else "no") for i in range(n)}


def test_sample_is_balanced_deterministic_and_seed_dependent():
    a = ho.sample_items(many_records(), CASES, verdicts(), n_per_class=5, seed=1)
    b = ho.sample_items(many_records(), CASES, verdicts(), n_per_class=5, seed=1)
    c = ho.sample_items(many_records(), CASES, verdicts(), n_per_class=5, seed=2)
    ids = lambda s: [i["finding_id"] for i, _ in s["judged"]]
    assert ids(a) == ids(b) and ids(a) != ids(c)
    assert sorted(v for _, v in a["judged"]).count("yes") == 5 and len(a["judged"]) == 10
    assert a["available"] == (10, 10, 4) and len(a["negative"]) == 4


def test_sample_takes_what_exists_when_a_class_is_short():
    s = ho.sample_items(many_records(), CASES, {"c1:positive:s0:f0": "yes", "c1:positive:s0:f1": "no",
                                                 "c1:positive:s0:f2": "yes"}, n_per_class=15)
    assert len(s["judged"]) == 3


def test_unjudged_and_other_split_findings_are_not_sampled():
    s = ho.sample_items(many_records(), {}, verdicts(), n_per_class=5)
    assert s["judged"] == [] and s["negative"] == []
    s = ho.sample_items(many_records(), CASES, {}, n_per_class=5)
    assert s["judged"] == []


def test_files_are_blinded_and_contain_the_patch(tmp_path):
    cases_dir = tmp_path / "cases"
    (cases_dir / "c1").mkdir(parents=True)
    (cases_dir / "c1" / "negative.diff").write_text("-old line\n+new line\n", encoding="utf-8")
    sampled = ho.sample_items(many_records(), CASES, verdicts(), n_per_class=3, seed=0)
    md, c1, c2 = ho.write_files(sampled, tmp_path / "out", cases_dir)
    text = md.read_text(encoding="utf-8")
    assert "+new line" in text and "msg c1:positive:s0:f" in text
    assert "verdict" not in text.lower() and "judge said" not in text.lower()
    rows = list(csv.DictReader(c1.open(encoding="utf-8")))
    assert len(rows) == 6 and set(rows[0]) == {"finding_id", "human_label"} and all(r["human_label"] == "" for r in rows)
    neg_rows = list(csv.DictReader(c2.open(encoding="utf-8")))
    assert set(neg_rows[0]) == {"finding_id", "real_issue"} and len(neg_rows) == 4


def test_read_labels_skips_blanks_and_rejects_bad_values(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("finding_id,human_label\na,Yes\nb,\nc, no \n", encoding="utf-8")
    assert ho.read_labels(p, "human_label", ("yes", "no")) == {"a": "yes", "c": "no"}
    p.write_text("finding_id,human_label\na,maybe\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ho.read_labels(p, "human_label", ("yes", "no"))


def test_kappa_hand_computed():
    pairs = [("yes", "yes"), ("yes", "no"), ("no", "no"), ("no", "no")]
    assert ho.cohens_kappa(pairs) == pytest.approx(0.5)        # po 0.75, pe 0.5
    assert ho.cohens_kappa([("yes", "yes")] * 4 + [("no", "no")] * 4) == pytest.approx(1.0)
    assert ho.cohens_kappa([("yes", "yes")] * 3) is None       # no variation: kappa undefined
    assert ho.cohens_kappa([]) is None


def test_agreement_counts_confusion_disagreements_and_missing():
    human = {"a": "yes", "b": "no", "c": "yes", "d": "no", "zz": "yes"}
    judged = {"a": "yes", "b": "yes", "c": "yes", "d": "no"}
    r = ho.agreement(human, judged)
    assert (r["n"], r["agree"], r["accuracy"]) == (4, 3, 0.75)
    assert r["confusion"] == {("yes", "yes"): 2, ("yes", "no"): 1, ("no", "yes"): 0, ("no", "no"): 1}
    assert r["disagreements"] == [("b", "yes", "no")] and r["missing"] == ["zz"]
    assert r["ci"] == pytest.approx(ho.mt.wilson(3, 4))


def test_cli_roundtrip_sample_then_agreement_then_latent(tmp_path, capsys):
    cases = labeling.load_cases(split="dev")
    loc = cases["httpie-4"]["bug_locations"][0]
    recs = [{"run_id": "httpie-4:positive:s0", "case_id": "httpie-4", "kind": "positive", "sample": 0, "status": "ok",
             "findings": [finding("httpie-4:positive:s0:f0", loc["start"], file=loc["file"]),
                          finding("httpie-4:positive:s0:f1", loc["start"], file=loc["file"])]},
            {"run_id": "httpie-4:negative:s0", "case_id": "httpie-4", "kind": "negative", "sample": 0, "status": "ok",
             "findings": [finding("httpie-4:negative:s0:f0", loc["start"], file=loc["file"])]}]
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps({"finding_id": k, "verdict": v, "status": "ok", "reason": "r",
                                            "prompt_version": judge.JUDGE_PROMPT_VERSION})
                                for k, v in {"httpie-4:positive:s0:f0": "yes", "httpie-4:positive:s0:f1": "no"}.items()),
                      encoding="utf-8")
    out = tmp_path / "hl"
    ho.main(["sample", str(runs), "--labels", str(labels), "--out-dir", str(out), "--n", "15"])
    printed = capsys.readouterr().out
    assert "sampled 2 judged + 1 negative-run" in printed and "NOTE: fewer than" in printed
    (out / "handlabel.csv").write_text("finding_id,human_label\nhttpie-4:positive:s0:f0,yes\n"
                                       "httpie-4:positive:s0:f1,yes\n", encoding="utf-8")
    ho.main(["agreement", "--csv", str(out / "handlabel.csv"), "--labels", str(labels)])
    text = capsys.readouterr().out
    assert "agreement 1/2 = 0.50" in text and "DISAGREE httpie-4:positive:s0:f1: judge no, you yes" in text
    (out / "negatives.csv").write_text("finding_id,real_issue\nhttpie-4:negative:s0:f0,unsure\n", encoding="utf-8")
    ho.main(["latent", "--csv", str(out / "negatives.csv")])
    assert "checked: 1   real issue 0, not an issue 0, unsure 1" in capsys.readouterr().out