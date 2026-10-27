"""Data that flows through the LangGraph pipeline."""
from enum import Enum
from typing import Optional, TypedDict

from pydantic import BaseModel, Field

from src.config import Category, Severity


class Decision(str, Enum):
    AUTO_POST = "auto_post"
    ESCALATE = "escalate"


class Finding(BaseModel):
    # --- from the reviewer LLM ---
    id: str
    file: str
    line: int
    category: Category
    severity: Severity
    message: str
    expected_failure: Optional[str] = None  # e.g. "AssertionError"; used to verify repro

    # --- signals: None means unavailable (see docs/signals.md) ---
    static_overlap: Optional[float] = Field(default=None, ge=0, le=1)
    sandbox_verified: Optional[float] = Field(default=None, ge=0, le=1)
    self_critique_agree: Optional[float] = Field(default=None, ge=0, le=1)

    # --- logged, not scored ---
    critic_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    critic_rebuttal: Optional[str] = None
    repro_code: Optional[str] = None
    sandbox_log: Optional[str] = None

    # --- gate outputs ---
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    effective_severity: Optional[Severity] = None  # after the security floor
    decision: Optional[Decision] = None
    reason: Optional[str] = None


class ReviewState(TypedDict):
    pr_ref: str                      # e.g. "owner/repo#12" or a local diff id
    diff: str                        # unified diff under review
    files: dict[str, str]            # path -> file contents
    static_issues: list[dict]        # raw ruff output
    findings: list[Finding]
    errors: list[str]                # non-fatal node failures

