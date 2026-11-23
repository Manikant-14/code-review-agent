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

## UPDATE 2026-11-14 (supersedes "Not yet in the repo", the Phase 3 step 1-2 items, "LLM situation" model plans and "Resume" above)
Repo main at 2e75bf5, 29 tests pass. Phase 3 steps 1-2 DONE.

### Done
- Provider layer committed: model ids are "provider:model" (gemini, groq). Groq is called over plain httpx
  (src/llm.py _HttpCompletions), NOT the openai SDK: this laptop's Windows Application Control policy blocks
  pip.exe and compiled packages (jiter). Always use "python -m pip". FAISS may be blocked later.
  Network drops (DNS, dropped connection) are retried; "request too large" is fatal.
- Smoke test: python -m scripts.smoke_llm groq:openai/gpt-oss-120b
- Bake-off: python -m benchmark.bakeoff <ids> (5 cases: httpie-4 thefuck-29 tornado-10 httpie-2 thefuck-10, positive
  and negative diff each; location-only scoring). 120b: valid 10/10, found 3/5 (really 4/5), quiet on fixes 2/5,
  ~1.35K tokens/call. Qwen 3.8 27B: found 2/5, quiet 0/5. gpt-oss-20b: valid 6/10 (Groq JSON validation fails). Gemma 31B:
  found 4/5, valid 9/10, 50-290 s per call, ~5-10% Google 500s.
- Critic/judge mini-test: python -m benchmark.critic_judge_test <ids> (7 real 120b findings: 4 true, 3 false).
  Judge: Gemma 6/6 answered correct, Qwen 4/7, Flash Lite 4/7 (they say "incorrect" almost always).
  Critic: Qwen 3/7, Flash Lite 3/7 (chance), Gemma 5/6 but Gemma is the judge so it cannot also be the critic.
- config.py: MODELS reviewer groq:openai/gpt-oss-120b (3 RPM), judge gemini:gemma-4-31b-it (6 RPM), critic None ON PURPOSE.

### Limits and numbers
- Groq free: 8K TPM / 200K TPD per model, reasoning tokens count (~3-4x visible output). 120b call ~1.35K tokens, so a 102-call
  reviewer pass ~140K tokens, ~35 min at 3 RPM. The old 2.4M-token estimate assumed 5K prompts; diff-only prompts are ~600 tokens.
- Gemma 4 31B and 26B: 30 RPM, 16K TPM, 14.4K RPD. Judge must run offline, resumable, ideally 4-8 parallel calls (step 6).
- Gemini 3.5 and 3.1 Flash Lite: 15 RPM, 250K TPM, 500 RPD (fast, but poor as critic/judge in the test). Other Gemini Flash: 20 RPD.
- Cache: data/llm_cache; log: data/llm_log.jsonl (role, model, cached, latency_s, prompt_tokens, output_tokens).

### Findings that change the plan
1. All four models flag fix diffs (negative cases): they treat removed lines as the intended baseline. evaluation.md counts these as
   false positives by default. DECIDE at step 6: keep, or also report a separate "fix-flag rate". Do not tune the prompt on 5 cases.
2. GROUND-TRUTH BUG: build_cases.py records only changed buggy lines and skips pure-insertion locations inside mixed hunks.
   Uncovered: thefuck-10, thefuck-24, thefuck-28, httpie-5 (httpie-2's is a comment, harmless). Fix BEFORE step 6.
3. Diff-only critic cannot verify claims and rebuts true findings (chance level). Supports the sandbox-heavy weights; ablations decide.
   Choose the critic in Phase 4 on the pilot with the real prompt (candidates: Qwen, Gemma 26B (same family as judge: partial
   independence), Flash Lite).
4. n is tiny (5 cases, 7 findings): screening only, not ranking.

### Next (Phase 3)
3. Review-input builder: promote annotate_diff (benchmark/bakeoff.py) to src; add bounded context via git show from data/repos;
   test with and without context on the same 5 cases; keep under ~5K tokens; log the deviation in evaluation.md.
4. Reviewer node (schema file, line, category, severity, message, expected_failure, self_confidence), temperature 0.2, JSONL output.
5. Thin skeleton then 10-bug pilot (20 runs). 6. Eval harness (first fix build_cases). 7. Baselines and calibration in evaluation.md.

### Working style
One step at a time with reasoning and a recommendation; every step as complete copy-paste PowerShell blocks (here-string Python
patchers with exact anchors, idempotent, tested on a fresh clone first). Free only, pinned model ids, never paste .env contents.

### Resume
  cd C:\dev\code-review-agent
  .\.venv\Scripts\Activate.ps1
  git pull
  python -m pytest -q      (29 passed)


## UPDATE 2026-11-21 (Phase 3 step 3)
Step 3a done (d82fa15, 42 tests): src/review_input.py builds the reviewer prompt (annotated diff + enclosing
function, capped), benchmark/context_test.py compares with/without context (arm A reuses the bake-off cache).
Result on 5 cases: context found 5/5 vs 3/5 (4/5 after the thefuck-10 ground-truth gap), 0 vs 1 extra, quiet on
fixes 3/5 vs 2/5. Context is the default; deviation logged in docs/evaluation.md.
Numbers: a context call is ~1.2K prompt + ~0.7K output tokens; use 2 RPM for the reviewer; a full 102-run pass is
~195K tokens, about the whole 200K/day Groq cap, so plan it across two days (cache makes it resumable).
Still open: fix-flagging on negative diffs (decide at step 6), build_cases ground-truth gap (fix before step 6).
Next: step 4, reviewer node (schema file, line, category, severity, message, expected_failure, self_confidence),
temperature 0.2, findings saved as JSONL.

## UPDATE 2026-11-22 (Phase 3 steps 4-5, ground truth)
Steps 4-5 done. Reviewer node src/reviewer.py (c577312): one JSONL line per run in data/runs/, resumable, prompt v1,
Finding.self_confidence added. Pilot (10 bugs, 20 runs, data/runs/pilot.jsonl): no errors, location recall 10/10
(weak signal: the bug is the diff), fix-flag 4/10 negative runs, self_confidence hits 0.89 vs other findings 0.85.
Ground-truth gap FIXED: build_cases.hunk_locations adds anchors for pure-insertion groups in mixed hunks;
benchmark/refresh_locations.py regenerates bug_locations from negative.diff (7 cases changed: httpie-5, thefuck-10,
thefuck-11, thefuck-13, thefuck-24, thefuck-28, tornado-5); a test keeps meta.json in sync. Logged in evaluation.md.
Next: step 6, eval harness per docs/evaluation.md (location matcher, judge only on location-matched findings, metrics,
intervals, unit tests). Decide the fix-flag reporting there.

## CURRENT STATE 2026-11-22 (read this first; supersedes older UPDATE sections where they conflict)
Phase 3 is complete except two human tasks (see Open). 153 tests pass.

How the pieces fit
- src/review_input.py: case -> prompt (annotated diff + enclosing function, capped; context on by default).
- src/reviewer.py: reviewer node, prompt v1, one JSONL line per run, resumable. Finding.self_confidence is a baseline feature only.
- benchmark/labeling.py: location matcher, labeling rules, dataset builder, judge queue.
- benchmark/judge.py: Gemma 4 31B judge, prompt j2, 4 parallel workers, resumable labels.jsonl. llm.py throttle and log are thread-safe.
- benchmark/metrics.py, benchmark/calibration.py: metrics, Wilson and cluster-bootstrap intervals, operating curve, Brier/ECE/AUROC.
- benchmark/evaluate.py: the report. benchmark/handlabel.py: blinded judge check (30 findings) and negative-run latent check.
- benchmark/refresh_locations.py: regenerates bug_locations from negative.diff; a test keeps meta.json in sync.

Commands (repo root, venv active)
  python -m src.reviewer CASE... --out data\runs\X.jsonl
  python -m benchmark.judge data\runs\X.jsonl
  python -m benchmark.evaluate data\runs\X.jsonl --policy score --param 0.9 --curve --calibration
  python -m benchmark.handlabel sample data\runs\X.jsonl      (then: agreement, latent)

Decisions made in Phase 3 (all logged in docs/evaluation.md)
- Reviewer input = diff + enclosing function (<= 80 lines per function, ~4000-token cap); full files dropped (Groq budget).
- Ground truth: pure-insertion groups in mixed hunks are anchors (7 cases refreshed).
- Judge prompt j2 (the reviewer sees the diff reversed); j2 is frozen for dev-set judging.
- Fix-flagging: negative-run findings stay false positives by default; the fix-flag rate is reported separately; no prompt tuning.
- Baselines (no gate, always escalate, self-confidence sweep, later critic-only) and calibration (Brier, ECE, AUROC)
  defined before any dev run. Bootstrap intervals that collapse at 0/1 are shown with Wilson.

Pilot (10 bugs, 20 runs, judge j2; plumbing only, wide intervals): 18 findings, 12 true positives; no-gate precision
0.67 [0.44, 0.84]; fix-flag rate 4/10; detection recall 10/10 (Wilson [0.72, 1.00]); self-confidence >= 0.9 gives
precision 0.86 [0.49, 0.97] but auto-post recall 0.60 and 61% escalation. False positives cluster at 0.85, so the
self-confidence baseline is a real opponent.

Open
- Manikant: hand-label data/runs/handlabel (14 judged + 4 negative-run findings), then run handlabel agreement and latent.
- The full dev pass (102 reviewer runs, ~195K tokens) needs two days of Groq budget, then ~70 judge calls (Phase 5).

Next: Phase 4, one node at a time on the pilot: 1 static (ruff on the full file, changed lines only);
2 sandbox (decide the per-project Docker image strategy first); 3 critic (different model family, binary verdict);
4 wire gate.py into LangGraph and persist every signal for offline replay.
Pilot calibration of self_confidence (18 findings, wide intervals): Brier 0.26 vs base-rate predictor 0.22
(skill -0.16 [-0.48, 0.02]); ECE 0.23 [0.08, 0.41]; AUROC 0.65 [0.40, 0.85]. Reliability: scores 0.80-0.85 (n=10)
were right 50% of the time, 0.88-0.92 (n=5) 80%, 0.93-0.95 (n=3) 100%. Reading: overconfident, and the pilot cannot
yet say whether self-confidence ranks better than chance (the AUROC interval includes 0.5). The earlier wording
"the self-confidence baseline is a real opponent" is a hypothesis for the dev run, not a finding.

## UPDATE 2026-11-23 (Phase 4 step 1: static node)
src/static_analysis.py: ruff via `python -m ruff` (--isolated --select F,E9) over the full new-version file on
stdin; issues kept only if their row span touches an added line; static_overlap = 1.0 within +/-3 lines of a
finding, else None. A ruff failure leaves the signal unavailable and is logged in state errors. Checked on this
laptop: ruff 0.17.2 and Docker 29.9.2 both run, so the pyflakes fallback was not needed. Rule set pre-registered
in evaluation.md ("Static analysis"); rule-set ablation planned for Phase 5.
Replay and persist for offline gate replay:
  python -m src.static_analysis data\runs\pilot.jsonl --out data\signals\static.jsonl
Next: Phase 4 step 2, sandbox node (decide the per-project Docker image strategy first).