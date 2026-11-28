"""Sandbox environment layer (Phase 4 step 2a): one Docker image per distinct bug environment, plus a runner.

Design (docs/evaluation.md "Sandbox environment"):
  * Image key = hash(python minor + the bug's BugsInPy requirements + tooling pins). The 51 dev cases share
    6 environments, so images are built lazily, once each, with the network on (mirrors benchmark/verify_repro.py).
  * Every run uses a throwaway tree from `git archive` of one commit, `--network none`, memory/CPU/pid limits and a
    hard kill at SANDBOX_TIMEOUT_SECONDS. Timeouts and setup errors are "unavailable", never a 0.0 signal.
  * Two modes: "cmd" runs a shell command (BugsInPy's own test, used only to validate the environment) and
    "cov" runs a pytest repro file under coverage and reports the executed lines per file.

CLI (repo root, venv active, Docker running):
  python -m src.sandbox check httpie-4 thefuck-29 ...
runs, per case: build/reuse the image, BugsInPy's test on the buggy and fixed trees (expect fail/pass as in
benchmark/repro_results), then a coverage + network-off self-test on the buggy tree.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src import config

ROOT = Path(__file__).resolve().parent.parent
BUGSINPY = ROOT / "data" / "BugsInPy" / "projects"
REPOS = ROOT / "data" / "repos"
SANDBOX_DIR = ROOT / "data" / "sandbox"
CASES_DIR = ROOT / "benchmark" / "cases"
REPRO_RESULTS = ROOT / "benchmark" / "repro_results"

ENV_VERSION = "1"
TOOLING = ("coverage==7.2.7", "pytest==7.4.4 (only if absent)")
REPRO_FILE = "sandbox_repro_test.py"
IMAGE_PREFIX = "crabench-env:"

BUILD_SCRIPT = r"""set -u
: > /opt/pip-failed.log
# One requirement per line so a single bad pin cannot abort the install (as BugsInPy and verify_repro do).
while IFS= read -r req; do
  pip install -q "$req" > /dev/null 2>&1 || echo "$req" >> /opt/pip-failed.log
done < /opt/reqs.txt
# jinja2 < 3.1 imports markupsafe.soft_unicode, removed in markupsafe 2.1
if pip show jinja2 > /dev/null 2>&1; then
  pip install -q "markupsafe<2.1" > /dev/null 2>&1 || echo "markupsafe<2.1" >> /opt/pip-failed.log
fi
pip install -q coverage==7.2.7 > /dev/null 2>&1 || echo "coverage==7.2.7" >> /opt/pip-failed.log
pip show pytest > /dev/null 2>&1 || pip install -q pytest==7.4.4 > /dev/null 2>&1 || echo "pytest==7.4.4" >> /opt/pip-failed.log
exit 0
"""

DOCKERFILE = "FROM python:{minor}\nCOPY reqs.txt build_env.sh /opt/\nRUN bash /opt/build_env.sh\n"

# Runs inside the container. SRC/OUT/PYTHON/LIMIT are overridable so the script can be tested without Docker.
RUN_SH = r"""#!/bin/bash
SRC="${SRC:-/src}"; OUT="${OUT:-/out}"; PY="${PYTHON:-python}"; LIMIT="${LIMIT:-30}"
cd "$SRC" || { echo 99 > "$OUT/rc"; exit 0; }
export COVERAGE_FILE="$OUT/.coverage"
if [ "${1:-cmd}" = "cov" ]; then
  timeout -s KILL "$LIMIT" "$PY" -m coverage run --source="$SRC" -m pytest -q -p no:cacheprovider "$SRC/sandbox_repro_test.py" > "$OUT/output.log" 2>&1
  rc=$?
  echo "$rc" > "$OUT/rc"
  "$PY" -m coverage json -q -o "$OUT/cov.json" >> "$OUT/output.log" 2>&1 || true
else
  timeout -s KILL "$LIMIT" bash "$OUT/cmd.sh" > "$OUT/output.log" 2>&1
  rc=$?
  echo "$rc" > "$OUT/rc"
fi
exit 0
"""


class SandboxError(Exception):
    """Docker or the environment is unusable: the signal is unavailable, not zero."""


# ---------------------------------------------------------------- environment spec

@dataclass
class EnvSpec:
    case_id: str
    python_minor: str
    requirements: list[str]
    pythonpath: list[str] = field(default_factory=list)
    run_test: str = ""
    test_files: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return env_key(self.python_minor, self.requirements)

    @property
    def tag(self) -> str:
        return IMAGE_PREFIX + self.key


def read_kv(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def normalize_requirements(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    return [ln for ln in lines if ln and not ln.startswith("#") and not ln.startswith("-e")]


def env_key(python_minor: str, requirements: list[str]) -> str:
    blob = "\n".join([ENV_VERSION, python_minor, *requirements, "--tooling--", *TOOLING])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def read_env_spec(case_id: str, bugsinpy_dir: Optional[Path] = None) -> EnvSpec:
    project, _, bug_id = case_id.rpartition("-")
    bug_dir = (bugsinpy_dir or BUGSINPY) / project / "bugs" / bug_id
    if not bug_dir.exists():
        raise SandboxError(f"no BugsInPy data for {case_id} at {bug_dir} (see STATUS.md: clone BugsInPy into data\\)")
    info = read_kv(bug_dir / "bug.info")
    req = bug_dir / "requirements.txt"
    return EnvSpec(
        case_id=case_id,
        python_minor=".".join(info["python_version"].split(".")[:2]),
        requirements=normalize_requirements(req.read_text(encoding="utf-8", errors="replace")) if req.exists() else [],
        pythonpath=[p.strip().strip("/") for p in info.get("pythonpath", "").split(";") if p.strip()],
        run_test=(bug_dir / "run_test.sh").read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n"),
        test_files=[t.strip() for t in info.get("test_file", "").split(";") if t.strip()],
    )


# ---------------------------------------------------------------- docker plumbing

def _docker(args: list[str], timeout: float, check_timeout_kill: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise SandboxError("docker is not installed or not on PATH") from e
    except subprocess.TimeoutExpired:
        if check_timeout_kill:
            subprocess.run(["docker", "rm", "-f", check_timeout_kill], capture_output=True, check=False)
        raise
    except OSError as e:
        raise SandboxError(f"docker could not run: {type(e).__name__}: {str(e)[:150]}") from e


def _text(b: bytes, n: int = 1500) -> str:
    return b.decode("utf-8", errors="replace")[-n:]


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def ensure_image(spec: EnvSpec, docker: Optional[Callable] = None, log: Callable[[str], None] = print) -> tuple[str, list[str]]:
    """Return (image tag, pip requirements that failed to install). Builds the image on first use (network on)."""
    docker = docker or _docker
    meta_path = SANDBOX_DIR / f"env_{spec.key}.json"
    r = docker(["image", "inspect", spec.tag], 60)
    if r.returncode == 0:
        failed = json.loads(meta_path.read_text(encoding="utf-8"))["pip_failed"] if meta_path.exists() else []
        return spec.tag, failed
    log(f"building {spec.tag} for python {spec.python_minor}, {len(spec.requirements)} requirements (first use, can take minutes)")
    ctx = SANDBOX_DIR / "build" / spec.key
    shutil.rmtree(ctx, ignore_errors=True)
    write_lf(ctx / "Dockerfile", DOCKERFILE.format(minor=spec.python_minor))
    write_lf(ctx / "build_env.sh", BUILD_SCRIPT)
    write_lf(ctx / "reqs.txt", "\n".join(spec.requirements) + "\n")
    r = docker(["build", "-q", "-t", spec.tag, str(ctx)], 3600)
    if r.returncode != 0:
        raise SandboxError(f"docker build failed for {spec.tag}: {_text(r.stderr)}")
    r = docker(["run", "--rm", "--network", "none", spec.tag, "cat", "/opt/pip-failed.log"], 120)
    failed = [ln.strip() for ln in r.stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    write_lf(meta_path, json.dumps({"key": spec.key, "tag": spec.tag, "python": spec.python_minor,
                                    "n_requirements": len(spec.requirements), "pip_failed": failed,
                                    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    return spec.tag, failed


# ---------------------------------------------------------------- trees

def extract_tree(repo: Path, commit: str, dest: Path, overlay_from: Optional[str] = None,
                 overlay_paths: tuple[str, ...] = ()) -> Path:
    """Fresh copy of one commit via `git archive` (no autocrlf, no .git). Optionally copy some files from another commit."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    r = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True)
    if r.returncode != 0:
        raise SandboxError(f"git archive {commit[:8]} failed: {_text(r.stderr, 300)}")
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        for m in tf.getmembers():
            if not (m.isfile() or m.isdir()):
                continue  # symlinks and special files are skipped on purpose
            target = (dest / m.name).resolve()
            if root != target and root not in target.parents:
                raise SandboxError(f"unsafe path in archive: {m.name}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(m).read())
    for path in overlay_paths:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{overlay_from}:{path}"], capture_output=True)
        if r.returncode != 0:
            raise SandboxError(f"git show {overlay_from[:8]}:{path} failed: {_text(r.stderr, 300)}")
        out = dest / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.stdout)
    return dest


# ---------------------------------------------------------------- running

@dataclass
class SandboxResult:
    status: str                      # pass | fail | timeout | error
    rc: Optional[int]
    seconds: float
    output_tail: str = ""
    executed: dict = field(default_factory=dict)   # path relative to the tree -> executed line numbers


def classify(rc: Optional[int], timed_out: bool = False) -> str:
    if timed_out or rc in (124, 137):
        return "timeout"
    if rc is None:
        return "error"
    if rc == 0:
        return "pass"
    if rc == 1:
        return "fail"          # pytest: tests ran and at least one failed
    return "error"             # 2 interrupted/collection error, 3 internal, 4 usage, 5 no tests, 99 setup, ...


def parse_coverage(cov_json_text: str) -> dict[str, list[int]]:
    data = json.loads(cov_json_text)
    out = {}
    for name, d in data.get("files", {}).items():
        rel = name.replace("\\", "/")
        if rel.startswith("/src/"):
            rel = rel[len("/src/"):]
        out[rel] = sorted(d.get("executed_lines", []))
    return out


def run_sandbox(tag: str, src_dir: Path, mode: str, command: Optional[str] = None, repro: Optional[str] = None,
                pythonpath: tuple[str, ...] = (), timeout: int = config.SANDBOX_TIMEOUT_SECONDS,
                docker: Optional[Callable] = None) -> SandboxResult:
    docker = docker or _docker
    if mode not in ("cmd", "cov"):
        raise ValueError(f"mode must be cmd or cov, got {mode!r}")
    out = SANDBOX_DIR / "runs" / uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    write_lf(out / "run.sh", RUN_SH)
    if mode == "cmd":
        write_lf(out / "cmd.sh", command or "true\n")
    else:
        write_lf(Path(src_dir) / REPRO_FILE, repro or "")
    name = "crabench-" + uuid.uuid4().hex[:10]
    pp = ":".join(["/src", *("/src/" + p for p in pythonpath)])
    args = ["run", "--rm", "--name", name, "--network", "none", "--memory", config.SANDBOX_MEMORY,
            "--cpus", config.SANDBOX_CPUS, "--pids-limit", str(config.SANDBOX_PIDS),
            "-v", f"{src_dir}:/src", "-v", f"{out}:/out", "-w", "/src",
            "-e", f"PYTHONPATH={pp}", "-e", f"LIMIT={timeout}", tag, "bash", "/out/run.sh", mode]
    t0 = time.time()
    try:
        r = docker(args, timeout + 90, check_timeout_kill=name)
    except subprocess.TimeoutExpired:
        return SandboxResult("timeout", None, time.time() - t0, "docker run exceeded the outer timeout")
    secs = time.time() - t0
    rc_file = out / "rc"
    rc = int(rc_file.read_text().strip()) if rc_file.exists() and rc_file.read_text().strip().lstrip("-").isdigit() else None
    log = out / "output.log"
    tail = log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.exists() else _text(r.stderr, 500)
    executed = {}
    cov = out / "cov.json"
    if mode == "cov" and cov.exists():
        try:
            executed = parse_coverage(cov.read_text(encoding="utf-8"))
        except (ValueError, TypeError):
            executed = {}
    return SandboxResult(classify(rc), rc, secs, tail, executed)


# ---------------------------------------------------------------- validation CLI

def module_name(path: str) -> str:
    mod = path[:-3] if path.endswith(".py") else path
    mod = mod.replace("\\", "/").replace("/", ".")
    return mod[: -len(".__init__")] if mod.endswith(".__init__") else mod


def selftest_repro(bug_file: str) -> str:
    return (
        "import importlib\nimport socket\n\nimport pytest\n\n\n"
        "def test_bug_module_imports():\n"
        f"    importlib.import_module({module_name(bug_file)!r})\n\n\n"
        "def test_network_is_off():\n"
        "    with pytest.raises(OSError):\n"
        "        socket.create_connection((\"1.1.1.1\", 53), timeout=3)\n"
    )


def check_case(case_id: str, docker: Optional[Callable] = None, log: Callable[[str], None] = print) -> dict:
    docker = docker or _docker
    spec = read_env_spec(case_id)
    meta = json.loads((CASES_DIR / case_id / "meta.json").read_text(encoding="utf-8"))
    expected = json.loads((REPRO_RESULTS / f"{case_id}.json").read_text(encoding="utf-8"))
    repo = REPOS / meta["project"]
    tag, failed = ensure_image(spec, docker, log)
    trees = SANDBOX_DIR / "trees"
    buggy = extract_tree(repo, meta["buggy_commit"], trees / f"{case_id}-buggy",
                         overlay_from=meta["fixed_commit"], overlay_paths=tuple(spec.test_files))
    fixed = extract_tree(repo, meta["fixed_commit"], trees / f"{case_id}-fixed")
    pp = tuple(spec.pythonpath)
    rb = run_sandbox(tag, buggy, "cmd", command=spec.run_test, pythonpath=pp, timeout=120, docker=docker)
    rf = run_sandbox(tag, fixed, "cmd", command=spec.run_test, pythonpath=pp, timeout=120, docker=docker)
    bug_file = meta["files"][0]
    rs = run_sandbox(tag, buggy, "cov", repro=selftest_repro(bug_file), pythonpath=pp, docker=docker)
    cov_lines = len(rs.executed.get(bug_file, []))
    env_ok = rb.rc not in (None, 0) and rb.status != "timeout" and rf.rc == 0
    rc_same = (rb.rc, rf.rc) == (expected.get("buggy_rc"), expected.get("fixed_rc"))
    self_ok = rs.status == "pass" and cov_lines > 0
    return {"case": case_id, "env": spec.key, "pip_failed": failed, "buggy": rb, "fixed": rf, "self": rs,
            "expected": (expected.get("buggy_rc"), expected.get("fixed_rc")), "env_ok": env_ok,
            "rc_same": rc_same, "cov_lines": cov_lines, "self_ok": self_ok}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "check":
        print(__doc__)
        return 2
    bad, seen_failed = 0, set()
    print(f"{'case':14s} {'env':12s} {'buggy rc (want)':>16s} {'fixed rc (want)':>16s} {'sec':>6s}  coverage/network self-test")
    for cid in argv[1:]:
        try:
            r = check_case(cid)
        except SandboxError as e:
            print(f"{cid:14s} SETUP ERROR: {e}")
            bad += 1
            continue
        for f in r["pip_failed"]:
            if (r["env"], f) not in seen_failed:
                seen_failed.add((r["env"], f))
                print(f"  note: env {r['env']} could not install: {f}")
        sec = r["buggy"].seconds + r["fixed"].seconds + r["self"].seconds
        st = f"{r['self'].status}, bug-file lines executed={r['cov_lines']}"
        verdict = "OK" if (r["env_ok"] and r["rc_same"] and r["self_ok"]) else "CHECK"
        print(f"{cid:14s} {r['env']:12s} {str(r['buggy'].rc) + ' (' + str(r['expected'][0]) + ')':>16s} "
              f"{str(r['fixed'].rc) + ' (' + str(r['expected'][1]) + ')':>16s} {sec:6.1f}  {st}  {verdict}")
        if verdict != "OK":
            bad += 1
            if not r["self_ok"]:
                print("    self-test output tail:\n" + "\n".join("      " + ln for ln in r["self"].output_tail.splitlines()[-12:]))
            if not r["env_ok"] or not r["rc_same"]:
                print("    buggy output tail:\n" + "\n".join("      " + ln for ln in r["buggy"].output_tail.splitlines()[-8:]))
    print(f"\n{len(argv) - 1 - bad} OK, {bad} need a look")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))