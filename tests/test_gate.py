import pytest

from src.config import Category, Severity
from src.gate import compute_confidence, decide
from src.state import Decision, Finding


def make(**kw) -> Finding:
    base = dict(
        id="f", file="a.py", line=1,
        category=Category.LOGIC, severity=Severity.MEDIUM, message="m",
    )
    base.update(kw)
    return Finding(**base)


def test_no_signals_escalates():
    r = decide(make())
    assert r.decision == Decision.ESCALATE
    assert r.reason == "no_signals"
    assert r.confidence is None


def test_sandbox_veto_overrides_everything():
    r = decide(make(static_overlap=1.0, sandbox_verified=0.0, self_critique_agree=1.0))
    assert r.decision == Decision.ESCALATE
    assert r.reason == "sandbox_veto"


def test_llm_only_evidence_never_auto_posts():
    r = decide(make(self_critique_agree=1.0))
    assert r.decision == Decision.ESCALATE
    assert r.reason == "no_non_llm_support"


def test_weights_renormalize_over_available_signals():
    assert compute_confidence({"static_overlap": 1.0, "self_critique_agree": 0.0}) == pytest.approx(0.4)
    assert compute_confidence({"sandbox_verified": 1.0, "self_critique_agree": 0.0}) == pytest.approx(0.625)


def test_all_signals_agree_auto_posts():
    r = decide(make(sandbox_verified=1.0, self_critique_agree=1.0))
    assert r.decision == Decision.AUTO_POST
    assert r.confidence == pytest.approx(1.0)


def test_verified_but_disputed_by_critic_escalates():
    r = decide(make(severity=Severity.LOW, sandbox_verified=1.0, self_critique_agree=0.0))
    assert r.decision == Decision.ESCALATE  # 0.625 < 0.65
    assert r.reason.startswith("below_threshold")


def test_low_severity_uses_lenient_threshold():
    r = decide(make(severity=Severity.LOW, static_overlap=1.0, sandbox_verified=1.0, self_critique_agree=0.0))
    assert r.decision == Decision.AUTO_POST  # 0.7 >= 0.65


def test_medium_severity_is_stricter_than_low():
    r = decide(make(severity=Severity.MEDIUM, static_overlap=1.0, sandbox_verified=1.0, self_critique_agree=0.0))
    assert r.decision == Decision.ESCALATE  # 0.7 < 0.75


def test_security_floor_raises_severity():
    r = decide(make(category=Category.SECURITY, severity=Severity.LOW,
                    static_overlap=1.0, sandbox_verified=1.0, self_critique_agree=0.0))
    assert r.effective_severity == Severity.HIGH
    assert r.decision == Decision.ESCALATE  # would auto-post as LOW (0.7 >= 0.65)


def test_decide_does_not_mutate_input():
    f = make(sandbox_verified=1.0, self_critique_agree=1.0)
    decide(f)
    assert f.decision is None and f.confidence is None
