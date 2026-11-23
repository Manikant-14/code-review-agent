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


## Deviations log
- 2026-11-21 (Phase 3 step 3): the reviewer sees the annotated diff plus the enclosing function of each
  hunk (at most 80 lines per function, whole prompt capped at ~4,000 estimated tokens, read with
  `git show` at the buggy commit for positive runs and the fixed commit for negative runs), not the
  full changed files named in "Unit of evaluation". Reason: the Groq free tier allows 8K tokens/min and
  200K tokens/day on the reviewer model, and reasoning tokens count. Screening on 5 dev cases,
  location-only: found 5/5 with context vs 3/5 without (4/5 after the known thefuck-10 ground-truth gap),
  extra findings 0 vs 1, quiet on fixes 3/5 vs 2/5. n = 5, so this is a screen, not a ranking; the
  with/without comparison should be repeated on the pilot with the judge in place.

- 2026-11-22 (Phase 3 step 5b, ground truth): bug_locations now include one anchor location for each
  pure-insertion change group inside a mixed hunk (code the buggy version is missing) that lies outside
  the removed range and is not comment-only. Before, such insertions were dropped, so a reviewer pointing
  at a missing check could be scored as a miss. Seven cases changed: httpie-5, thefuck-10, thefuck-11,
  thefuck-13, thefuck-24, thefuck-28, tornado-5. A fix to the ground truth made before any scoring run;
  location matching is only a pre-filter before the judge, so the extra anchors cannot make a wrong finding
  count as correct. benchmark/refresh_locations.py regenerates locations from negative.diff and a test keeps
  meta.json in sync.
- 2026-11-22 (Phase 3 step 6b, judge): judge prompt j1 -> j2. j1 answered no to a finding whose wording was
  reversed relative to the patch: the reviewer sees the fixed->buggy diff, so it says "removed" where the
  patch adds code (pilot case tornado-4 f2). j2 adds one sentence telling the judge about the reversal; the
  yes/no rule is unchanged. The change was made after the pilot judge run, from the judge's own stated
  reason, not from hit counts. On the 12 pilot findings judged under both prompts, 1 verdict(s) changed (tornado-4:positive:s0:f2);
  the rest were identical. j1 verdicts stay in labels.jsonl and are ignored. j2 is frozen for dev-set
  judging; agreement with hand labels is the validation.



## Baselines and calibration (added 2026-11-22, before any dev-set run)
Baselines, each scored with the same labels and intervals as the system:
- No gate: every finding is auto-posted (the reviewer alone).
- Always escalate: a human reviews everything and nothing is auto-posted; the reference for escalation cost.
- Reviewer self-confidence: auto-post when the reviewer's own self_confidence is at or above a threshold;
  the full threshold sweep is reported as an operating curve. This is the baseline the gate must beat; treat
  it as a real opponent, not a strawman.
- Critic only: auto-post when the critic's confidence is at or above a threshold (once the critic node exists,
  Phase 4).
- Token log-probabilities: only for a provider that returns them; none is planned, so this baseline is dropped
  unless one is found.
Thresholds behind any headline number are not tuned on the data they are reported on (grouped cross-validation
by bug, Phase 5); curves show every threshold.

Calibration and ranking of a confidence score (self-confidence, critic confidence, gate confidence before and
after the calibrator), over all scored findings, positive and negative runs, with label = true positive as
defined above:
- Brier score, with the base-rate predictor (always predict the overall true-positive rate) as reference and
  the skill score 1 - Brier/reference.
- Expected calibration error over up to 5 equal-mass bins, identical scores never split across bins (scores
  cluster on a few values), with the reliability table (bin range, count, mean score, observed true-positive
  rate).
- AUROC: the probability that a random true positive scores above a random false positive, ties counting half.
  It isolates ranking quality, which a threshold gate depends on, from calibration, which Brier and ECE also mix in.
Intervals: cluster bootstrap over cases (B = 2000, fixed seed), as for the other run-level metrics.

Interval clarification: a percentile bootstrap collapses to a single point when every case succeeds or fails,
which is not certainty. Run-level proportions (detection recall, auto-post recall, fix-flag rate) are therefore
also reported with a Wilson interval (valid with one repeat per case), and the operating curve shows the Wilson
interval wherever its bootstrap band has zero width.

## Static analysis (added 2026-11-23, before any dev-set run)
- Engine and rules: `python -m ruff check --isolated --select F,E9` (pyflakes-class correctness rules plus
  syntax/IO errors). `--isolated` ignores the benchmark project's own ruff config, so results do not depend on it.
  Style and convention rules (E4, E7, W, C90 ...) are excluded on purpose: static_overlap is a non-LLM signal that
  can unlock an auto-post, so a style hit near an unrelated finding must not count as corroboration.
- Input: the full new-version file at the commit the reviewer saw (buggy commit for positive runs, fixed commit
  for negative runs). An issue is kept only if its row span touches a line the diff added.
- Overlap: a finding gets static_overlap = 1.0 if a kept issue in the same file has a row span intersecting
  [line - 3, line + 3]; otherwise the signal is unavailable (never 0.0). Non-Python files, and files that cannot be
  read or linted (ruff blocked, timeout), are unavailable and logged in the run's errors.
- The ruff version is stored in every signal record (data/signals/static.jsonl). A rule-set ablation (ruff default
  rules; F+E9+B) is planned for Phase 5 and is reported next to the headline numbers.