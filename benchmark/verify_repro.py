"""Check that each BugsInPy bug reproduces: its test fails on buggy code, passes on fixed code.

Buggy version = buggy commit + test file(s) from the fixed commit (mirrors BugsInPy's
own checkout script). Fixed version = fixed commit.

Usage:  python -m benchmark.verify_repro thefuck 1 2 3    (specific bugs)
        python -m benchmark.verify_repro thefuck          (every bug in the project)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUGSINPY = ROOT / "data" / "BugsInPy" / "projects"
REPOS = ROOT / "data" / "repos"
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "benchmark" / "repro_results"
TIMEOUT_S = 1800


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def git(*args):
    r = sh(["git", "-c", "core.autocrlf=false", *args])
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def read_kv(path):
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def prepare(project, bug_id, info, github_url):
    cache = REPOS / project
    if not cache.exists():
        REPOS.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", github_url, str(cache))
    tag = f"{project}-{bug_id}"
    buggy, fixed = WORK / f"{tag}-buggy", WORK / f"{tag}-fixed"
    for dest, commit in ((buggy, info["buggy_commit_id"]), (fixed, info["fixed_commit_id"])):
        if not dest.exists():
            WORK.mkdir(parents=True, exist_ok=True)
            git("clone", "--quiet", "--no-checkout", str(cache), str(dest))
            git("-C", str(dest), "checkout", "--quiet", commit)
    for tf in filter(None, (t.strip() for t in info["test_file"].split(";"))):
        git("-C", str(buggy), "checkout", "--quiet", info["fixed_commit_id"], "--", tf)
    return buggy, fixed


def runner_script(info):
    extra = "".join(
        f":/$v/{p.strip().strip('/')}"
        for p in info.get("pythonpath", "").split(";")
        if p.strip()
    )
    # Mirrors BugsInPy: install requirements one line at a time so a single bad pin
    # (or the editable install of the project itself) cannot abort the whole install.
    return rf"""set -u
: > /out/pip.log
sed -e 's/\r$//' -e '/^\s*#/d' -e '/^\s*$/d' -e '/^-e/d' /bug/requirements.txt > /out/reqs.txt
while IFS= read -r req; do
  pip install -q "$req" >> /out/pip.log 2>&1 || echo "FAILED: $req" >> /out/pip.log
done < /out/reqs.txt
# jinja2 < 3.1 imports markupsafe.soft_unicode, removed in markupsafe 2.1
if pip show jinja2 > /dev/null 2>&1; then pip install -q "markupsafe<2.1" >> /out/pip.log 2>&1 || echo "FAILED: markupsafe pin" >> /out/pip.log; fi
for v in buggy fixed; do
  cd /$v
  PYTHONPATH="/$v{extra}" timeout 600 bash /out/test_cmd.sh > /out/$v.log 2>&1
  echo $? > /out/$v.rc
done
"""


def run_in_docker(info, bug_dir, buggy, fixed, out):
    out.mkdir(parents=True, exist_ok=True)
    cmd_text = (bug_dir / "run_test.sh").read_text(encoding="utf-8", errors="replace")
    (out / "test_cmd.sh").write_text(cmd_text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    (out / "runner.sh").write_text(runner_script(info), encoding="utf-8", newline="\n")
    image = "python:" + ".".join(info["python_version"].split(".")[:2])
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{buggy}:/buggy", "-v", f"{fixed}:/fixed",
        "-v", f"{bug_dir}:/bug:ro", "-v", f"{out}:/out",
        "-v", "bugsinpy-pipcache:/root/.cache/pip",
        image, "bash", "/out/runner.sh",
    ]
    return sh(cmd, timeout=TIMEOUT_S)


def read_rc(out, v):
    p = out / f"{v}.rc"
    return int(p.read_text().strip()) if p.exists() else None


def tail(out, v, n=400):
    p = out / f"{v}.log"
    return p.read_text(errors="replace")[-n:] if p.exists() else ""


def verify(project, bug_id):
    bug_dir = BUGSINPY / project / "bugs" / str(bug_id)
    info = read_kv(bug_dir / "bug.info")
    github_url = read_kv(BUGSINPY / project / "project.info")["github_url"]
    out = WORK / f"{project}-{bug_id}-out"
    result = {
        "project": project, "bug_id": int(bug_id), "split": "dev",
        "python_version": info.get("python_version"), "test_file": info.get("test_file"),
    }
    try:
        buggy, fixed = prepare(project, bug_id, info, github_url)
        r = run_in_docker(info, bug_dir, buggy, fixed, out)
        result["buggy_rc"], result["fixed_rc"] = read_rc(out, "buggy"), read_rc(out, "fixed")
        result["reproduces"] = result["buggy_rc"] not in (None, 0) and result["fixed_rc"] == 0
        result["buggy_tail"] = tail(out, "buggy")  # eyeball: is it the real bug, not an env error?
        if result["buggy_rc"] is None:
            result["error"] = r.stderr[-500:]
    except Exception as e:  # noqa: BLE001
        result["reproduces"] = False
        result["error"] = str(e)[:500]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{project}-{bug_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    project, *ids = sys.argv[1:]
    if not ids:
        ids = sorted((p.name for p in (BUGSINPY / project / "bugs").iterdir()), key=int)
    for b in ids:
        res = verify(project, b)
        print(project, b, "OK" if res["reproduces"] else "NO", res.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
