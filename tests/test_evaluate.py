"""Tests for benchmark/evaluate.py on real case metadata with synthetic runs and verdicts."""
import json

import pytest

from benchmark import evaluate as ev, judge, labeling

HTTPIE = labeling.load_cases(split="dev")["httpie-4"]["bug_locations"][0]


def finding(fid, line, file, score=0.9, **kw):
    d = dict(id=fid, file=file, line=line, category="logic", severity="medium", message="m",
             expected_failure=None, self_confidence=score)
    d.update(kw)
    return d


def rec(cid, kind, findings, sample=0, status="ok"):
    return {"run_id": f"{cid}:{kind}:s{sample}", "case_id": cid, "kind": kind, "sample": sample, "status": status,
            "error": None, "findings": findings, "est_tokens": 100}


def write(tmp_path, records, verdicts):
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps({"finding_id": k, "case_id": k.split(":")[0], "verdict": v, "reason": "r",
                                            "status": "ok", "prompt_version": judge.JUDGE_PROMPT_VERSION,
                                            "error": None}) for k, v in verdicts.items()), encoding="utf-8")
    return runs, labels


def records():
    f = HTTPIE["file"]
    return [
        rec("httpie-4", "positive", [finding("httpie-4:positive:s0:f0", HTTPIE["start"], f),
                                     finding("httpie-4:positive:s0:f1", HTTPIE["start"] + 300, f, score=0.5)]),
        rec("httpie-4", "negative", [finding("httpie-4:negative:s0:f0", HTTPIE["start"], f)]),
        rec("thefuck-10", "positive", []),
        rec("thefuck-10", "negative", []),
    ]


def run_main(tmp_path, capsys, verdicts=None, extra=()):
    runs, labels = write(tmp_path, records(), {"httpie-4:positive:s0:f0": "yes"} if verdicts is None else verdicts)
    ev.main([str(runs), "--labels", str(labels), "--boot", "200", *extra])
    return capsys.readouterr().out


def test_no_gate_report_numbers(tmp_path, capsys):
    out = run_main(tmp_path, capsys)
    assert "runs used 4" in out and "findings 3: true positive 1, false positive 2" in out
    assert "judge_yes 1" in out and "location 1" in out and "negative_run 1" in out
    line = lambda label: next(l for l in out.splitlines() if label in l)
    assert "0.33 [" in line("auto-post precision") and "wilson" in line("auto-post precision")
    assert line("escalation rate").split()[2] == "0.00"
    assert "0.50 [" in line("detection recall") and "bootstrap" in line("detection recall")
    assert "0.50 [" in line("fix-flag rate")


def test_judge_no_makes_the_location_match_a_false_positive(tmp_path, capsys):
    out = run_main(tmp_path, capsys, {"httpie-4:positive:s0:f0": "no"})
    assert "true positive 0" in out and "judge_no 1" in out
    assert "detection recall                           0.00" in out


def test_pending_judge_blocks_the_report(tmp_path, capsys):
    runs, labels = write(tmp_path, records(), {})
    with pytest.raises(SystemExit) as e:
        ev.main([str(runs), "--labels", str(labels)])
    assert "waiting for the judge" in str(e.value) and "benchmark.judge" in str(e.value)


def test_score_policy_block_and_curve(tmp_path, capsys):
    out = run_main(tmp_path, capsys, extra=["--policy", "score", "--param", "0.85", "--curve"])
    assert "SCORE THRESHOLD BASELINE (auto-post when score >= 0.85)" in out
    # the 0.5-confidence false positive is escalated: auto-posted 2 of 3 findings
    assert "auto-posted 2, escalated 1" in out
    assert "OPERATING CURVE (score" in out and "1.0" in out


def test_policy_needs_a_param(tmp_path):
    runs, labels = write(tmp_path, records(), {})
    with pytest.raises(SystemExit):
        ev.main([str(runs), "--policy", "score"])


def test_other_split_runs_are_skipped_and_reported(tmp_path, capsys):
    recs = records() + [rec("not-a-case", "positive", [])]
    runs, labels = write(tmp_path, recs, {"httpie-4:positive:s0:f0": "yes"})
    ev.main([str(runs), "--labels", str(labels), "--boot", "50"])
    assert "skipped 1" in capsys.readouterr().out


def test_spread_block_appears_only_with_repeats(tmp_path, capsys):
    base = records()
    s1 = [rec(r["case_id"], r["kind"], [dict(f, id=f["id"].replace(":s0:", ":s1:")) for f in r["findings"]], sample=1)
          for r in base]
    runs, labels = write(tmp_path, base + s1, {"httpie-4:positive:s0:f0": "yes", "httpie-4:positive:s1:f0": "no"})
    ev.main([str(runs), "--labels", str(labels), "--boot", "50"])
    out = capsys.readouterr().out
    assert "SPREAD ACROSS 2 REPEATS" in out
    assert "detection_recall" in out and "0.00 / 0.25 / 0.50" in out


def test_no_spread_block_with_a_single_repeat(tmp_path, capsys):
    assert "SPREAD" not in run_main(tmp_path, capsys)


def test_formatting_helpers():
    assert ev.fmt(None) == "n/a" and ev.fmt(0.5) == "0.50"
    assert ev.fmt_entry({"value": 0.5, "ci": (0.25, 0.75)}) == "0.50 [0.25, 0.75]"
    assert ev.fmt_entry({"value": None, "ci": None}) == "n/a"