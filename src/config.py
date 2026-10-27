"""Central configuration. Every value here traces back to docs/signals.md."""
from enum import Enum


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Signal weights (renormalized at runtime over whichever signals are available).
SIGNAL_WEIGHTS = {
    "static_overlap": 0.2,
    "sandbox_verified": 0.5,
    "self_critique_agree": 0.3,
}

# At least one of these must be available and positive to auto-post.
NON_LLM_SIGNALS = ("static_overlap", "sandbox_verified")

# Minimum confidence to auto-post, by severity.
AUTO_POST_THRESHOLDS = {
    Severity.LOW: 0.65,
    Severity.MEDIUM: 0.75,
    Severity.HIGH: 0.90,
}

# Rule-based severity floor: these categories are always treated as HIGH,
# whatever severity the reviewer LLM assigned.
HIGH_SEVERITY_CATEGORIES = frozenset({"security", "injection", "auth", "secrets"})

# static_overlap: ruff must flag a line within +/- this many lines of the finding.
STATIC_OVERLAP_LINE_WINDOW = 3

# Sandbox run limit before the result is marked unavailable.
SANDBOX_TIMEOUT_SECONDS = 30

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"
