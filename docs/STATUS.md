# Project Status and Handoff

Last updated: 2026-11-09. Read this first when resuming. Design source of truth:
docs/problem_statement.md, docs/signals.md, docs/evaluation.md. Older chat-memory notes may
mention a superseded design (0.7 formula with severity_prior); trust the docs.

## Project
Self-Correcting Code Review Agent with Confidence-Gated Human Escalation. College project
(Sharda University PBL-3, CSP-391). Publication plan: book chapter first, then a full paper;
also a patent description of the gating function. Evaluation rubrics: R2 was scheduled for
2026-11-03; R3 (results review) comes later.
Core question: when is a single review finding trustworthy enough to auto-post, and when must
it escalate to a human? Goals (not promises): auto-posted precision > 90%, escalation < 40%.

## Working rules
- Windows + PowerShell. Repo C:\dev\code-review-agent, github.com/Manikant-14/code-review-agent
  (main). Activate venv: .\.venv\Scripts\Activate.ps1
- Must stay free of cost (no billing anywhere). No GPU on the laptop (Ryzen AI 5 340, NPU only).
- One step at a time: explain reasoning, give options with a recommendation, avoid code dumps,
  proceed strictly in phase order, commit after each step.
- API keys live in .env (git-ignored). Never paste keys into chat.

## Design (final unless noted)
Pipeline (LangGraph, 7 nodes): ingestion -> static analysis (ruff on the full file, filtered to
changed lines) -> LLM review -> sandbox (Docker) -> self-critique -> confidence gate ->
auto-post / escalate.
Signals (0..1, or None = unavailable):
- static_overlap: 1.0 if ruff flags a line within +/-3 of the finding, else unavailable.
- sandbox_verified: 1.0 repro fails as predicted; 0.0 repro runs clean and coverage confirms
  the flagged line ran; else unavailable.
- self_critique_agree: 1.0 no concrete rebuttal, 0.0 rebuttal found. Critic runs on a different
  model; critic confidence and rebuttal are logged, not scored.
Weights: static 0.2, sandbox 0.5, critique 0.3, renormalised over available signals.
Gate rules, in order: no signals -> escalate; sandbox == 0.0 -> veto, escalate; no available
positive non-LLM signal -> escalate; confidence < severity threshold -> escalate; else auto-post.
Thresholds: low 0.65, medium 0.75, high 0.90. Category "security" forces HIGH.
Categories: logic, security, error_handling, concurrency, performance, style.
State: one shared state with findings: list[Finding] (Pydantic). Switch to LangGraph Send
fan-out only if it becomes a bottleneck.
Evaluation (docs/evaluation.md): a finding is correct if its location is within +/-3 lines of
any bug location AND a judge model (different from reviewer and critic) says it describes the
patched defect; judge audited by hand on ~30 findings. Findings on negative runs are false
positives by default. Metrics: auto-post precision, escalation rate (escalated findings / all
findings), detection recall, auto-post recall, false-positive load, precision-vs-escalation
curve; Wilson intervals and cluster bootstrap by case; CV grouped by bug.

## Done (commits)
- Setup and problem statement: 8877e54, 19f0aa3, 7a5889c
- Phase 1 design: signals.md fbf5b83, config.py 07ce5e6, state.py 4f86b28
- Phase 1.5: gate.py + 10 tests aa53372
- Phase 2 benchmark: verify_repro + results 4c6513d, build_cases 627c739, LF pinning c5aa1d0,
  evaluation.md 35c16a5, location tests d1d904f, pytest.ini 3510086
- Phase 3 started: cached rate-limited LLM client (Gemini only) 20cedbb, gitignore 774dbd2
- 23 tests pass.
Benchmark: BugsInPy dev set = thefuck, tornado, httpie (Python 3.7). 53 bugs reproduce
(test fails on buggy, passes on fixed; stable over 3 runs for tornado/httpie). 2 dropped (test
patches a symbol the fix adds), leaving 51 cases (thefuck 31, tornado 16, httpie 4). Each
benchmark/cases/<id>/ has positive.diff (fixed->buggy), negative.diff (buggy->fixed) and
meta.json (bug locations on the buggy side; 13 missing-code anchors, 14 multi-location;
category still null). Upstream data lives in data/ (git-ignored). Rebuild:
  git clone --depth 1 https://github.com/soarsmu/BugsInPy.git data\BugsInPy
  python -m benchmark.verify_repro thefuck   (and tornado, httpie)
  python -m benchmark.build_cases
Python 3.7 Docker image is used for verification; BugsInPy tests are ground truth only and are
never shown to the reviewer or repro generator.

## LLM situation (free only)
- Gemini free tier: Flash models ~5 RPM, ~20 requests/day seen. Gemma 4 31B: 30 RPM, 16K TPM,
  daily limit unknown (check the AI Studio rate-limit page, RPD column).
- Groq free plan: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b (preview), each
  30 RPM, 1K RPD, 8K TPM, 200K TPD, 131K context. Llama 3.3 70B / 3.1 8B are Enterprise-only.
- Local LLM judged impractical as the backbone (CPU/NPU only). Lemonade Server on the NPU may be
  tried later as a judge/overflow after a 30-minute speed/context test. Colab/Kaggle GPU is plan B.
- Leading plan: three model families, reviewer groq:openai/gpt-oss-120b, critic
  groq:qwen/qwen3.8-27b, judge Gemma 4 31B via the Gemini API; confirm with a 5-case bake-off.
  MODELS in config.py are still None. Never use -latest aliases; pin exact ids.
- Estimated full pass ~800 calls / ~2.4M tokens (a guess, to be measured; reasoning tokens may
  double it), about 4-5 days across three Groq models; a 20-run pilot fits in about a day. The
  disk cache makes runs resumable; archive data/llm_cache with the paper artifacts.
- Reviewer prompts must stay under ~5K tokens (8K TPM), so the review context is the diff plus
  the enclosing function / nearby lines instead of whole files. Log this as a deviation in
  docs/evaluation.md.

## Not yet in the repo
- Provider layer ("provider:model" ids, OpenAI-compatible adapter for Groq/xAI/OpenRouter/
  Ollama, python-dotenv loading, tests/test_llm_providers.py). Spec: config.PROVIDERS table;
  llm._split, _gemini_call, _openai_call returning the same {text, usage_metadata} shape as the
  Gemini path; JSON mode via response_format json_object plus the schema in the system text.
- openai and pytest in requirements.txt; a README.

## Roadmap and how
Phase 3 (now)
1. Apply and commit the provider layer; smoke-test one real call (groq:openai/gpt-oss-120b).
2. 5-case bake-off of candidate models; fill MODELS in config.py; set RPM_LIMITS.
3. Review-input builder: from a case dir, diff (strip "index" lines) + bounded context via
   git show from data/repos; keep under ~5K tokens.
4. Reviewer node: schema (file, line, category, severity, message, expected_failure,
   self_confidence for the baseline), temperature 0.2, calls through llm.call_llm, findings
   saved to JSONL.
5. Thin skeleton: ingest one case -> reviewer -> print; then a 10-bug pilot (20 runs).
6. Eval harness implementing docs/evaluation.md: location matcher, judge only on
   location-matched findings, metrics, intervals, unit tests.
7. Extend evaluation.md with baselines (reviewer self-rated confidence, token logprobs if
   available, critic-only, no gate, always-escalate) and calibration (ECE, Brier, reliability).
Phase 4 (one node at a time, each tested on the pilot)
1. Static node: ruff on the full file at the relevant commit, filter to changed lines.
2. Sandbox node: per-project prebuilt Docker images, network off, 30 s timeout; an LLM writes a
   repro using expected_failure; coverage check; produces sandbox_verified. Per-bug dependency
   pins differ, so decide the image strategy first.
3. Self-critique node: different model family, strongest-counterargument prompt, binary verdict.
4. Wire gate.py into LangGraph; persist every signal for offline replay.
Phase 5: full dev pass; trained calibrator (grouped CV by bug); ablations (drop each signal,
with/without the non-LLM rule, flat vs severity thresholds, formula vs calibrator); baselines;
operating curves; assign categories to the 51 cases; hand-audit 30 judge labels; add planted
security cases to test the floor; add held-out test projects once (e.g. tqdm, black, PySnooper).
Phase 6: GitHub App webhook (FastAPI), MongoDB escalation queue, Streamlit dashboard, Docker
Compose; keep every service on free tiers.
Phase 7: book chapter, then paper; patent description (check IP-cell advice and timing).

## Open decisions and risks
- Final model choice for the three roles (needs the Gemma RPD and the bake-off).
- Escalation < 40% may be unreachable if few repros succeed; report the operating curve.
- Few distinct confidence values with binary signals: medium and high thresholds behave alike
  until the calibrator exists. static_overlap is weak corroboration (test in ablations).
- 51 positive cases give wide intervals. Public bugs may be memorised by models; diffs are tiny.
- The repo is public; patent timing needs advice (consider making it private).

## Resume
  cd C:\dev\code-review-agent
  .\.venv\Scripts\Activate.ps1
  git pull
  python -m pytest -q      (23 passed; 26 after the provider tests)
