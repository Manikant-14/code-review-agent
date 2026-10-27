"""Confidence gate: findings in, decisions out. Pure functions, no I/O.

Rules are applied in a fixed order (see docs/signals.md); the first one that
fires sets `reason`, so escalations can be counted per rule in the ablations.
"""
from typing import Optional

from src.config import (
    AUTO_POST_THRESHOLDS,
    HIGH_SEVERITY_CATEGORIES,
    NON_LLM_SIGNALS,
    SIGNAL_WEIGHTS,
    Severity,
)
from src.state import Decision, Finding

EPS = 1e-9  # float tolerance for threshold comparison


def available_signals(f: Finding) -> dict[str, float]:
    """Signals that are not None (None means unavailable)."""
    return {n: getattr(f, n) for n in SIGNAL_WEIGHTS if getattr(f, n) is not None}


def compute_confidence(signals: dict[str, float]) -> Optional[float]:
    """Weighted average over available signals only (weights renormalized)."""
    if not signals:
        return None
    total = sum(SIGNAL_WEIGHTS[n] for n in signals)
    return sum(SIGNAL_WEIGHTS[n] * v for n, v in signals.items()) / total


def effective_severity(f: Finding) -> Severity:
    """Reviewer's severity, forced to HIGH for floor categories."""
    return Severity.HIGH if f.category in HIGH_SEVERITY_CATEGORIES else f.severity


def _apply_rules(
    signals: dict[str, float], confidence: Optional[float], severity: Severity
) -> tuple[Decision, str]:
    if not signals:
        return Decision.ESCALATE, "no_signals"
    if signals.get("sandbox_verified") == 0.0:
        return Decision.ESCALATE, "sandbox_veto"
    if not any(signals.get(n, 0.0) > 0 for n in NON_LLM_SIGNALS):
        return Decision.ESCALATE, "no_non_llm_support"
    threshold = AUTO_POST_THRESHOLDS[severity]
    if confidence + EPS < threshold:
        return Decision.ESCALATE, f"below_threshold({confidence:.3f}<{threshold})"
    return Decision.AUTO_POST, "passed_gate"


def decide(f: Finding) -> Finding:
    """Return a copy of the finding with gate outputs filled in."""
    signals = available_signals(f)
    confidence = compute_confidence(signals)
    severity = effective_severity(f)
    decision, reason = _apply_rules(signals, confidence, severity)
    return f.model_copy(
        update={
            "confidence": confidence,
            "effective_severity": severity,
            "decision": decision,
            "reason": reason,
        }
    )


def gate_all(findings: list[Finding]) -> list[Finding]:
    return [decide(f) for f in findings]
