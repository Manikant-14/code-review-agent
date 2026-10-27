# Problem Statement

## The problem
LLM code reviewers can hallucinate issues. Auto-posting every review risks
false claims on real pull requests. Requiring a human to check every review
removes the time savings of automation.

## Core question
How does an automated reviewer decide, per finding, when it is trustworthy
enough to post without a human, and when it must escalate?

## Hypothesis
A confidence score built from independent evidence (static analysis,
sandboxed execution, adversarial self-critique) predicts review correctness
better than the model's own self-rating or raw token probabilities.

## Success criteria (measurable)
1. Escalation rate: share of findings sent to humans (target: below 40%).
2. Precision of auto-posted findings (target: above 90%).
3. Recall: share of benchmark bugs caught overall.
4. Calibration: confidence scores match actual correctness (ECE).

## Out of scope (for now)
- Languages other than Python
- Auto-fixing code (we only review)
- Replacing human reviewers entirely

## Deliverables
Working GitHub-integrated agent, benchmark dataset, ablation study,
paper draft, patent description of the gating function.
