"""Review-input builder: turns a benchmark case into the text the reviewer sees.

The prompt is the annotated diff (new-file line numbers, no index/header noise) plus, optionally,
bounded context: the function that encloses each hunk, read with `git show <commit>:<path>` from
data/repos/<project>. The positive run reads the buggy commit, the negative run the fixed commit,
so context line numbers always match the diff's numbers. The BugsInPy test is never read here.
"""
from __future__ import annotations

import ast
import json
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from unidiff import PatchSet

from src import config

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "benchmark" / "cases"
REPOS_DIR = ROOT / "data" / "repos"
PROMPT_HEAD = "Review this change.\n\n"


def annotate_diff(diff_text: str) -> str:
    """Render a diff with new-file line numbers; index lines are dropped."""
    out = []
    for pf in PatchSet(diff_text):
        out.append(f"=== {pf.path}")
        for hunk in pf:
            out.append(f"@@ new lines {hunk.target_start}-{hunk.target_start + hunk.target_length - 1} @@")
            for line in hunk:
                text = line.value.rstrip("\n")
                if line.is_removed:
                    out.append(f"    -| {text}")
                elif line.is_added:
                    out.append(f"{line.target_line_no:5d}+| {text}")
                else:
                    out.append(f"{line.target_line_no:5d} | {text}")
    return "\n".join(out)


def estimate_tokens(text: str) -> int:
    return int(len(text) / config.CHARS_PER_TOKEN) + 1


def git_show(project: str, commit: str, path: str, repos_dir: Optional[Path] = None) -> Optional[str]:
    """File contents at a commit, or None if the repo, commit or path is missing."""
    repo = (repos_dir or REPOS_DIR) / project
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                           capture_output=True, check=False)
    except OSError:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def function_spans(source: str) -> list[tuple[int, int]]:
    """(start, end) 1-based inclusive for every function/method, decorators included."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old projects have invalid escape sequences
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            spans.append((start, node.end_lineno))
    return spans


def innermost(spans: list[tuple[int, int]], line: int) -> Optional[tuple[int, int]]:
    best = None
    for s, e in spans:
        if s <= line <= e and (best is None or e - s < best[1] - best[0]):
            best = (s, e)
    return best


def _clip(span: tuple[int, int], hs: int, he: int, max_lines: int) -> tuple[int, int]:
    """Keep a long function to max_lines, centred on the part that overlaps the hunk."""
    s, e = span
    if e - s + 1 <= max_lines:
        return s, e
    centre = (max(hs, s) + min(he, e)) // 2
    ns = max(s, centre - max_lines // 2)
    ne = min(e, ns + max_lines - 1)
    ns = max(s, ne - max_lines + 1)
    return ns, ne


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def context_ranges(source: str, hunks: list[tuple[int, int]], max_func_lines: int,
                   radius: int) -> list[tuple[int, int]]:
    """Line ranges (new version) to show for one file: the enclosing function of each hunk."""
    n_lines = len(source.splitlines())
    spans = function_spans(source)
    ranges = []
    for hs, he in hunks:
        found = {x for x in (innermost(spans, hs), innermost(spans, he)) if x}
        if found:
            ranges += [_clip(sp, hs, he, max_func_lines) for sp in found]
        else:
            ranges.append((hs - radius, he + radius))
    clamped = [(max(1, s), min(n_lines, e)) for s, e in ranges]
    return _merge([r for r in clamped if r[0] <= r[1]])


def render_context(path: str, source: str, ranges: list[tuple[int, int]]) -> str:
    lines = source.splitlines()
    out = []
    for s, e in ranges:
        out.append(f"=== context: {path} (new version, lines {s}-{e}) ===")
        out += [f"{n:5d} | {lines[n - 1]}" for n in range(s, e + 1)]
    return "\n".join(out)


@dataclass
class ReviewInput:
    case_id: str
    kind: str
    text: str
    with_context: bool
    context_lines: int = 0
    est_tokens: int = 0
    over_budget: bool = False
    notes: list[str] = field(default_factory=list)


def _hunks_by_file(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for pf in PatchSet(diff_text):
        if pf.is_removed_file or pf.is_added_file:
            continue  # nothing to add: the diff already is the whole new file or no new file
        for h in pf:
            out.setdefault(pf.path, []).append(
                (h.target_start, h.target_start + max(h.target_length, 1) - 1))
    return out


def build_review_input(case_id: str, kind: str, with_context: bool = True,
                       cases_dir: Optional[Path] = None,
                       read_file: Optional[Callable[[str], Optional[str]]] = None) -> ReviewInput:
    """kind is "positive" (fixed->buggy diff, buggy commit) or "negative" (buggy->fixed, fixed commit).

    read_file(path) -> contents at the right commit; defaults to `git show` from data/repos.
    """
    if kind not in ("positive", "negative"):
        raise ValueError(f"kind must be positive or negative, got {kind!r}")
    case_dir = (cases_dir or CASES_DIR) / case_id
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    diff = (case_dir / f"{kind}.diff").read_text(encoding="utf-8")
    base = PROMPT_HEAD + annotate_diff(diff)
    ri = ReviewInput(case_id, kind, base, False, 0, estimate_tokens(base))
    ri.over_budget = ri.est_tokens > config.REVIEW_MAX_INPUT_TOKENS
    if ri.over_budget:
        ri.notes.append("diff alone exceeds the token budget")
    if not with_context:
        return ri
    commit = meta["buggy_commit"] if kind == "positive" else meta["fixed_commit"]
    if read_file is None:
        def read_file(path: str, _p=meta["project"], _c=commit) -> Optional[str]:
            return git_show(_p, _c, path)
    sources = {}
    for path in _hunks_by_file(diff):
        src = read_file(path)
        if src is None:
            ri.notes.append(f"no source for {path} at {commit[:8]}: context skipped")
        else:
            sources[path] = src
    if not sources:
        return ri
    hunks = _hunks_by_file(diff)
    full = config.REVIEW_MAX_FUNC_LINES
    for max_lines in (full, full // 2, full // 4, full // 8):
        radius = max(3, config.REVIEW_FALLBACK_RADIUS * max_lines // full)
        blocks, n = [], 0
        for path, src in sources.items():
            rs = context_ranges(src, hunks[path], max_lines, radius)
            n += sum(e - s + 1 for s, e in rs)
            blocks.append(render_context(path, src, rs))
        text = base + "\n\n" + "\n\n".join(b for b in blocks if b)
        toks = estimate_tokens(text)
        if toks <= config.REVIEW_MAX_INPUT_TOKENS:
            if max_lines != full:
                ri.notes.append(f"context shrunk to {max_lines} lines per function to fit the budget")
            return ReviewInput(case_id, kind, text, True, n, toks, False, ri.notes)
    ri.notes.append("context dropped: even the smallest context exceeds the budget")
    return ri