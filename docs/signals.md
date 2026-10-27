# Confidence Signals

Confidence = weighted evidence that a finding is *correct*. Each signal is in [0, 1]
or "unavailable". Unavailable signals are dropped and the remaining weights are
renormalized. Severity is NOT a signal; it adjusts the gate threshold.

## 1. static_overlap
- Measures: ruff flags a line within a few lines of the finding.
- Values: 1.0 if overlap, otherwise unavailable (never 0.0).
- Why: ruff is blind to logic bugs, so no overlap is not evidence against a finding.
- Limit: only covers what lint rules can see.

## 2. sandbox_verified
- Measures: an LLM-written repro, run in Docker, demonstrates the bug.
- 1.0: repro fails AND the failure matches the predicted one.
- 0.0: repro runs cleanly AND coverage confirms the flagged line executed.
- Unavailable: setup errors, timeouts, mismatched failure, line not covered,
  or a non-executable finding.
- Why: strongest signal; ground truth, independent of the reviewer's reasoning.

## 3. self_critique_agree
- Measures: an adversarial second pass tries to refute the finding.
- 1.0: no concrete rebuttal found. 0.0: concrete rebuttal found.
- Runs on a different model/tier than the reviewer.
- Logged but unused in the formula: critic confidence and rebuttal text
  (calibrator features, ablation, shown to the human on escalation).
- Limit: same model family as the reviewer, so errors are correlated.

## Severity (gate input, not a signal)
- Higher severity means a stricter auto-post threshold.
- Confidence answers "is it right?"; severity answers "how costly is it if we're wrong?"

## Gate rules
- Veto: sandbox_verified == 0.0 -> always escalate.
- No signals available -> escalate; no score is computed.
- Otherwise auto-post requires confidence >= severity-adjusted threshold AND
  at least one non-LLM signal (static_overlap or sandbox_verified) available and positive.
