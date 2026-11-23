"""Central configuration. Every value here traces back to docs/signals.md."""
from enum import Enum


class Category(str, Enum):
    LOGIC = "logic"
    SECURITY = "security"
    ERROR_HANDLING = "error_handling"
    CONCURRENCY = "concurrency"
    PERFORMANCE = "performance"
    STYLE = "style"


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
HIGH_SEVERITY_CATEGORIES = frozenset({Category.SECURITY})

# static_overlap: ruff must flag a line within +/- this many lines of the finding.
STATIC_OVERLAP_LINE_WINDOW = 3
# Rules ruff runs with (--isolated, so project config is ignored): pyflakes-class correctness rules plus
# syntax/IO errors. Style rules are excluded on purpose, see docs/evaluation.md "Static analysis".
STATIC_RULES = ("F", "E9")
STATIC_RUFF_TIMEOUT_SECONDS = 30

# Sandbox run limit before the result is marked unavailable.
SANDBOX_TIMEOUT_SECONDS = 30

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"

# --- Evaluation (docs/evaluation.md) ---
LOCATION_MATCH_WINDOW = 3

# --- LLM providers (Phase 3) ---
# Model ids are "provider:model", split on the FIRST colon (so "ollama:llama3:8b" also works).
# kind "gemini" uses the google-genai SDK; kind "openai" is any OpenAI-compatible endpoint.
# Keys are read from the environment / .env (never committed). Adding OpenRouter, xAI or a
# local Ollama later is one more entry here.
PROVIDERS = {
    "gemini": {"kind": "gemini"},
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY"},
}

# --- LLM roles (Phase 3) ---
# Pin exact model IDs (never "-latest" aliases: they change silently). The three roles must
# be three different models. None = not chosen yet.
# Reviewer chosen from the 5-case bake-off (valid JSON 10/10, ~1.35K tokens per call).
# Judge from the critic/judge mini-test: Gemma answered 6/6 correctly (Qwen and Flash Lite 4/7).
# Critic stays None on purpose: no tested model was reliable with diff-only context, so it is
# decided in Phase 4 on the pilot with the real critic prompt.
MODELS = {"reviewer": "groq:openai/gpt-oss-120b", "critic": None, "judge": "gemini:gemma-4-31b-it"}
TEMPERATURES = {"reviewer": 0.2, "critic": 0.0, "judge": 0.0}
# Requests per minute per model id (see AI Studio > Rate Limit); DEFAULT_RPM if not listed.
# Groq free tier is limited by tokens per minute (8K), not requests: 3 RPM leaves headroom.
RPM_LIMITS = {"groq:openai/gpt-oss-120b": 3, "gemini:gemma-4-31b-it": 6}
DEFAULT_RPM = 4

_chosen = [m for m in MODELS.values() if m]
assert len(set(_chosen)) == len(_chosen), "reviewer, critic and judge must be different models"
assert all(m.partition(":")[0] in PROVIDERS for m in _chosen), "model ids must be provider:model"


# --- Review input (Phase 3 step 3) ---
# Groq's free tier allows 8K tokens/min and reasoning tokens count, so the reviewer prompt is the
# diff plus bounded context (the enclosing function), not whole files. Deviation from the
# "full changed files" wording in docs/evaluation.md; logged there.
CHARS_PER_TOKEN = 3.5            # conservative estimate for source code
REVIEW_MAX_INPUT_TOKENS = 4000   # user prompt only; the system prompt adds a few hundred more
REVIEW_MAX_FUNC_LINES = 80       # longest stretch of one enclosing function that is shown
REVIEW_FALLBACK_RADIUS = 20      # lines either side of a hunk when no function encloses it