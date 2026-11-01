# Evaluation Protocol

Fixed before any results are seen. Changing a definition later must be logged as a deviation.

## Unit of evaluation
- A **run** = the reviewer applied to one review input: the `positive.diff` or the `negative.diff`
  of a case (plus full changed files via `git show`). One case therefore yields two runs.
- A run produces zero or more **findings**. Each finding passes through the signals and the gate
  and ends as `auto_post` or `escalate`.

## When a finding is correct (labeling)
A finding on a **positive** run is a true positive only if BOTH hold:
1. **Location:** same file as a `bug_locations` entry, and its line is within
   [start - W, end + W] of that entry. W = 3 (`LOCATION_MATCH_WINDOW`). A finding matching ANY
   location of the bug counts. Anchor-only locations (missing code) use start = end = anchor.
2. **Description:** a judge model answers yes to "does this finding describe the defect fixed by
   this patch?", given the finding text and the buggy->fixed patch only.

Everything else is a false positive. Findings on a **negative** run are false positives by default.

Judge rules:
- Different model from both the reviewer and the critic.
- Sees the finding text and the ground-truth patch. Never sees confidence, signals, or the gate decision.
- Validation: hand-label a random sample of 30 findings (15 judged yes, 15 judged no) and report
  judge agreement. Hand-check sampled negative-run findings for real latent issues.

## Metrics (all computed on the dev or test split separately, never pooled)
- **Auto-post precision** = true positives among auto-posted findings / auto-posted findings.
  Goal: > 90%.
- **Escalation rate** = escalated findings / all findings (positive and negative runs pooled).
  Goal: < 40%. Also reported separately for positive and negative runs.
- **Detection recall** = positive runs with at least one true-positive finding (any decision)
  / positive runs.
- **Auto-post recall** = positive runs with at least one auto-posted true positive / positive runs.
  This is the yield that needs no human.
- **False-positive load** = findings per negative run, split by auto-posted and escalated.
- **Operating curve:** auto-post precision vs escalation rate as the threshold scale is swept.
  Targets are goals, not promises; the curve is the headline result.

Duplicate findings on the same location within one run each count as findings for precision and
escalation, and count once toward run-level recall.

## Uncertainty
- Proportions over findings: Wilson 95% interval.
- Run-level and curve metrics: bootstrap over **cases** (cluster bootstrap), because findings from
  the same case are correlated. Positive and negative runs of one bug stay together.
- LLM nondeterminism: fixed temperature; repeat each run up to 3 times where budget allows and
  report the spread.

## Splits and leakage rules
- **Dev:** BugsInPy thefuck, tornado, httpie (Python 3.7). Used to build, tune, and train the
  calibrator.
- **Test:** held-out projects and Python versions, added later, evaluated once.
- Cross-validation on dev is grouped by **bug**: the positive and negative runs of one bug must be
  in the same fold. Where possible also group by project.
- The BugsInPy test files are ground truth for validation only. The reviewer, critic, and repro
  generator never see them.

## Threats to validity (to state in the paper)
- BugsInPy bugs are public; the models may have seen them.
- Patches are tiny (median 7 changed lines) compared to real PRs.
- Labels rely on location plus an LLM judge; agreement with hand labels is reported.
- Negative runs may contain real latent issues the dataset does not know about.
- Python only; one model family for reviewer and critic unless stated otherwise.
