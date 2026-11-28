import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src import config, sandbox as sb

HAS_GIT = shutil.which("git") is not None
try:
    import coverage  # noqa: F401
    HAS_COV = True
except ImportError:
    HAS_COV = False
CAN_RUN_SH = sys.platform != "win32" and shutil.which("bash") is not None and HAS_COV


@pytest.fixture(autouse=True)
def sandbox_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "SANDBOX_DIR", tmp_path / "sandbox")
    return tmp_path / "sandbox"


# ---- environment spec and key

def test_normalize_requirements_drops_comments_blanks_editable_and_crlf():
    text = "# c\r\nattrs==19.3.0\r\n\r\n-e .\r\n  click==7.1.2  \r\n"
    assert sb.normalize_requirements(text) == ["attrs==19.3.0", "click==7.1.2"]


def test_env_key_depends_on_python_and_requirements_only():
    a = sb.env_key("3.7", ["x==1", "y==2"])
    assert a == sb.env_key("3.7", ["x==1", "y==2"])
    assert a != sb.env_key("3.8", ["x==1", "y==2"])
    assert a != sb.env_key("3.7", ["x==1", "y==3"])
    assert len(a) == 12


def test_read_env_spec(tmp_path):
    bug = tmp_path / "proj" / "bugs" / "7"
    bug.mkdir(parents=True)
    (bug / "bug.info").write_text('python_version="3.7.3"\ntest_file="tests/a.py;tests/b.py"\npythonpath="pkg/sub/"\n', encoding="utf-8")
    (bug / "requirements.txt").write_text("attrs==1\n-e .\n", encoding="utf-8")
    (bug / "run_test.sh").write_bytes(b"pytest tests/a.py\r\n")
    s = sb.read_env_spec("proj-7", tmp_path)
    assert (s.python_minor, s.requirements, s.pythonpath) == ("3.7", ["attrs==1"], ["pkg/sub"])
    assert s.test_files == ["tests/a.py", "tests/b.py"] and s.run_test == "pytest tests/a.py\n"
    assert s.tag == "crabench-env:" + s.key


def test_read_env_spec_missing_data_is_a_clear_error(tmp_path):
    with pytest.raises(sb.SandboxError, match="no BugsInPy data"):
        sb.read_env_spec("proj-9", tmp_path)


def test_build_files_are_per_line_and_tolerant():
    assert "while IFS= read -r req" in sb.BUILD_SCRIPT and "pip-failed.log" in sb.BUILD_SCRIPT
    assert "markupsafe<2.1" in sb.BUILD_SCRIPT and "coverage==7.2.7" in sb.BUILD_SCRIPT
    assert sb.DOCKERFILE.format(minor="3.7").startswith("FROM python:3.7\n")


# ---- small pure helpers

@pytest.mark.parametrize("rc,timed,want", [(0, False, "pass"), (1, False, "fail"), (2, False, "error"), (5, False, "error"),
                                           (99, False, "error"), (None, False, "error"), (124, False, "timeout"),
                                           (137, False, "timeout"), (0, True, "timeout")])
def test_classify(rc, timed, want):
    assert sb.classify(rc, timed) == want


def test_parse_coverage_normalises_paths():
    raw = json.dumps({"files": {"/src/pkg/a.py": {"executed_lines": [3, 1]}, "pkg\\b.py": {"executed_lines": []}}})
    assert sb.parse_coverage(raw) == {"pkg/a.py": [1, 3], "pkg/b.py": []}


@pytest.mark.parametrize("path,mod", [("httpie/models.py", "httpie.models"), ("thefuck/rules/x.py", "thefuck.rules.x"),
                                      ("pkg/__init__.py", "pkg")])
def test_module_name(path, mod):
    assert sb.module_name(path) == mod


def test_selftest_repro_is_valid_python():
    compile(sb.selftest_repro("httpie/models.py"), "x", "exec")


# ---- trees

@pytest.mark.skipif(not HAS_GIT, reason="git not available")
def test_extract_tree_overlay_and_symlink_skip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", *a], check=True, capture_output=True)
    git("init", "-q")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "m.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "t.py").write_text("old\n", encoding="utf-8")
    git("add", "."); git("commit", "-qm", "one")
    first = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (repo / "pkg" / "m.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "tests" / "t.py").write_text("new\n", encoding="utf-8")
    try:
        os.symlink("m.py", repo / "pkg" / "link.py")
    except (OSError, NotImplementedError):
        pass
    git("add", "."); git("commit", "-qm", "two")
    second = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dest = tmp_path / "tree"
    sb.extract_tree(repo, first, dest, overlay_from=second, overlay_paths=("tests/t.py",))
    assert (dest / "pkg" / "m.py").read_text() == "x = 1\n"          # code from the first commit
    assert (dest / "tests" / "t.py").read_text() == "new\n"          # test overlaid from the second
    assert not (dest / ".git").exists()
    sb.extract_tree(repo, second, dest)                               # idempotent: dest is rebuilt
    assert (dest / "pkg" / "m.py").read_text() == "x = 2\n" and not (dest / "pkg" / "link.py").exists()


@pytest.mark.skipif(not HAS_GIT, reason="git not available")
def test_extract_tree_bad_commit_raises(tmp_path):
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    with pytest.raises(sb.SandboxError, match="git archive"):
        sb.extract_tree(repo, "deadbeef", tmp_path / "t")


# ---- run_sandbox with a fake docker

def fake_docker(rc=0, output="ok", cov=None, raises=None, record=None):
    def run(args, timeout, check_timeout_kill=None):
        if record is not None:
            record.append((args, timeout, check_timeout_kill))
        if raises:
            raise raises
        out = next(Path(a.rsplit(":/out", 1)[0]) for a in args if a.endswith(":/out"))
        if rc is not None:
            (out / "rc").write_text(f"{rc}\n")
        (out / "output.log").write_text(output)
        if cov is not None:
            (out / "cov.json").write_text(json.dumps(cov))
        return subprocess.CompletedProcess(args, 0, b"", b"")
    return run


def test_run_sandbox_locks_down_the_container(tmp_path):
    rec = []
    sb.run_sandbox("img", tmp_path, "cmd", command="pytest\n", pythonpath=("pkg",), docker=fake_docker(record=rec))
    args, timeout, kill = rec[0]
    j = " ".join(args)
    assert "--network none" in j and f"--memory {config.SANDBOX_MEMORY}" in j and "--pids-limit" in j and "--rm" in j
    assert f"LIMIT={config.SANDBOX_TIMEOUT_SECONDS}" in j and "PYTHONPATH=/src:/src/pkg" in j
    assert kill and kill.startswith("crabench-") and timeout > config.SANDBOX_TIMEOUT_SECONDS


def test_run_sandbox_statuses_and_coverage(tmp_path):
    cov = {"files": {"/src/m.py": {"executed_lines": [2, 5]}}}
    r = sb.run_sandbox("img", tmp_path, "cov", repro="def test_x(): pass\n", docker=fake_docker(rc=0, cov=cov))
    assert (r.status, r.rc, r.executed) == ("pass", 0, {"m.py": [2, 5]})
    assert (tmp_path / sb.REPRO_FILE).read_text() == "def test_x(): pass\n"
    assert sb.run_sandbox("img", tmp_path, "cmd", command="x", docker=fake_docker(rc=1, output="FAILED")).status == "fail"
    assert sb.run_sandbox("img", tmp_path, "cmd", command="x", docker=fake_docker(rc=137)).status == "timeout"
    r = sb.run_sandbox("img", tmp_path, "cmd", command="x", docker=fake_docker(rc=None, output="boom"))
    assert (r.status, r.rc, r.output_tail) == ("error", None, "boom")


def test_run_sandbox_outer_timeout_kills_container(tmp_path):
    r = sb.run_sandbox("img", tmp_path, "cmd", command="x", docker=fake_docker(raises=subprocess.TimeoutExpired("docker", 1)))
    assert r.status == "timeout"


def test_run_sandbox_rejects_bad_mode(tmp_path):
    with pytest.raises(ValueError):
        sb.run_sandbox("img", tmp_path, "nope")


# ---- ensure_image with a fake docker

def test_ensure_image_reuses_existing_and_builds_missing(sandbox_dir):
    spec = sb.EnvSpec("p-1", "3.7", ["attrs==1"])
    calls = []
    def docker(args, timeout, check_timeout_kill=None):
        calls.append(args[:2])
        if args[0] == "image":
            return subprocess.CompletedProcess(args, 1 if len([c for c in calls if c[0] == "image"]) == 1 else 0, b"", b"")
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, b"brotlipy==0.7.0\n", b"")
        return subprocess.CompletedProcess(args, 0, b"sha256:abc", b"")
    tag, failed = sb.ensure_image(spec, docker, log=lambda m: None)
    assert tag == spec.tag and failed == ["brotlipy==0.7.0"]
    ctx = sandbox_dir / "build" / spec.key
    assert (ctx / "reqs.txt").read_bytes() == b"attrs==1\n" and b"\r" not in (ctx / "Dockerfile").read_bytes()
    assert json.loads((sandbox_dir / f"env_{spec.key}.json").read_text())["pip_failed"] == ["brotlipy==0.7.0"]
    n = len(calls)
    assert sb.ensure_image(spec, docker, log=lambda m: None) == (spec.tag, ["brotlipy==0.7.0"])  # exists now: no build
    assert [c[0] for c in calls[n:]] == ["image"]


def test_ensure_image_build_failure_raises(sandbox_dir):
    spec = sb.EnvSpec("p-1", "3.7", [])
    def docker(args, timeout, check_timeout_kill=None):
        return subprocess.CompletedProcess(args, 1, b"", b"no space left")
    with pytest.raises(sb.SandboxError, match="no space left"):
        sb.ensure_image(spec, docker, log=lambda m: None)


# ---- the in-container script, run for real (no Docker): timeout, rc capture, coverage

def run_script(tmp_path, mode, repro=None, command=None, limit=30):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "pkg").mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    (src / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (src / "pkg" / "m.py").write_text("def f(x):\n    if x:\n        return 1\n    return 2\n", encoding="utf-8")
    (out / "run.sh").write_text(sb.RUN_SH, encoding="utf-8", newline="\n")
    if repro is not None:
        (src / sb.REPRO_FILE).write_text(repro, encoding="utf-8")
    if command is not None:
        (out / "cmd.sh").write_text(command, encoding="utf-8", newline="\n")
    env = {**os.environ, "SRC": str(src), "OUT": str(out), "PYTHON": sys.executable, "LIMIT": str(limit),
           "PYTHONPATH": str(src)}
    subprocess.run(["bash", str(out / "run.sh"), mode], env=env, check=True, capture_output=True, timeout=60)
    return int((out / "rc").read_text()), out


@pytest.mark.skipif(not CAN_RUN_SH, reason="needs bash and coverage (skipped on Windows)")
def test_script_cov_pass_reports_executed_lines(tmp_path):
    rc, out = run_script(tmp_path, "cov", repro="from pkg import m\n\ndef test_f():\n    assert m.f(1) == 1\n")
    assert rc == 0
    ex = sb.parse_coverage((out / "cov.json").read_text())
    assert 3 in ex["pkg/m.py"] and 4 not in ex["pkg/m.py"]   # branch x truthy ran, the fallthrough did not


@pytest.mark.skipif(not CAN_RUN_SH, reason="needs bash and coverage (skipped on Windows)")
def test_script_cov_fail_is_rc_1(tmp_path):
    rc, _ = run_script(tmp_path, "cov", repro="from pkg import m\n\ndef test_f():\n    assert m.f(0) == 1\n")
    assert sb.classify(rc) == "fail"


@pytest.mark.skipif(not CAN_RUN_SH, reason="needs bash and coverage (skipped on Windows)")
def test_script_collection_error_is_error_not_fail(tmp_path):
    rc, _ = run_script(tmp_path, "cov", repro="import does_not_exist\n\ndef test_f():\n    pass\n")
    assert sb.classify(rc) == "error"


@pytest.mark.skipif(not CAN_RUN_SH, reason="needs bash and coverage (skipped on Windows)")
def test_script_timeout_kills_hard(tmp_path):
    rc, _ = run_script(tmp_path, "cov", repro="import time\n\ndef test_f():\n    time.sleep(30)\n", limit=2)
    assert sb.classify(rc) == "timeout"


@pytest.mark.skipif(not CAN_RUN_SH, reason="needs bash and coverage (skipped on Windows)")
def test_script_cmd_mode_returns_command_rc(tmp_path):
    assert run_script(tmp_path, "cmd", command="exit 3\n")[0] == 3
    assert run_script(tmp_path, "cmd", command="true\n")[0] == 0


# ---- glue: check_case and the CLI against a synthetic project, fake docker

@pytest.mark.skipif(not HAS_GIT, reason="git not available")
def test_check_case_and_cli_end_to_end(tmp_path, monkeypatch, capsys):
    def git(repo, *a):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", *a],
                              check=True, capture_output=True, text=True).stdout.strip()
    repo = tmp_path / "repos" / "demo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "m.py").write_text("x = 'FIXED'\n", encoding="utf-8")
    (repo / "t.py").write_text("test-from-fix\n", encoding="utf-8")
    git(repo, "add", "."); git(repo, "commit", "-qm", "fixed")
    fixed = git(repo, "rev-parse", "HEAD")
    (repo / "pkg" / "m.py").write_text("x = 'BUG'\n", encoding="utf-8")
    (repo / "t.py").write_text("stale-test\n", encoding="utf-8")   # at the buggy commit the real test does not exist yet
    git(repo, "commit", "-qam", "buggy")
    buggy = git(repo, "rev-parse", "HEAD")
    bug = tmp_path / "bi" / "demo" / "bugs" / "1"
    bug.mkdir(parents=True)
    (bug / "bug.info").write_text('python_version="3.7.0"\ntest_file="t.py"\n', encoding="utf-8")
    (bug / "run_test.sh").write_text("pytest t.py\n", encoding="utf-8")
    case = tmp_path / "cases" / "demo-1"
    case.mkdir(parents=True)
    (case / "meta.json").write_text(json.dumps({"project": "demo", "buggy_commit": buggy, "fixed_commit": fixed,
                                                "files": ["pkg/m.py"]}), encoding="utf-8")
    res = tmp_path / "res"
    res.mkdir()
    (res / "demo-1.json").write_text(json.dumps({"buggy_rc": 1, "fixed_rc": 0}), encoding="utf-8")
    monkeypatch.setattr(sb, "BUGSINPY", tmp_path / "bi")
    monkeypatch.setattr(sb, "REPOS", tmp_path / "repos")
    monkeypatch.setattr(sb, "CASES_DIR", tmp_path / "cases")
    monkeypatch.setattr(sb, "REPRO_RESULTS", res)

    def docker(args, timeout, check_timeout_kill=None):
        if args[0] == "image":
            return subprocess.CompletedProcess(args, 0, b"", b"")   # image already built
        out = Path(next(a for a in args if a.endswith(":/out"))[:-5])
        src = Path(next(a for a in args if a.endswith(":/src"))[:-5])
        mode = args[-1]
        buggy_tree = "BUG" in (src / "pkg" / "m.py").read_text()
        (out / "rc").write_text("1\n" if (mode == "cmd" and buggy_tree) else "0\n")
        (out / "output.log").write_text("")
        if mode == "cov":
            (out / "cov.json").write_text(json.dumps({"files": {"/src/pkg/m.py": {"executed_lines": [1]}}}))
        return subprocess.CompletedProcess(args, 0, b"", b"")
    monkeypatch.setattr(sb, "_docker", docker)
    r = sb.check_case("demo-1", docker=docker, log=lambda m: None)
    assert r["env_ok"] and r["rc_same"] and r["self_ok"] and r["cov_lines"] == 1
    assert (tmp_path / "sandbox" / "trees" / "demo-1-buggy" / "t.py").read_text() == "test-from-fix\n"
    assert sb.main(["check", "demo-1"]) == 0
    assert "OK" in capsys.readouterr().out
    assert sb.main(["check", "demo-9"]) == 1     # unknown case: reported, not a crash