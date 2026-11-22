"""Thread-safety of the rate limiter and the call log (needed for parallel judge workers)."""
import json
import threading
import time

from src import llm


def test_throttle_gives_concurrent_callers_separate_slots(monkeypatch):
    monkeypatch.setitem(llm.RPM_LIMITS, "thr-model", 300)      # gap = 0.21 s
    monkeypatch.setattr(llm, "_last_call", {})
    stamps = []

    def worker():
        llm._throttle("thr-model")
        stamps.append(time.time())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) == 6 and min(gaps) >= 0.15              # without the lock they would fire together


def test_throttle_first_call_does_not_wait(monkeypatch):
    monkeypatch.setitem(llm.RPM_LIMITS, "thr-model-2", 6)
    monkeypatch.setattr(llm, "_last_call", {})
    t0 = time.time()
    llm._throttle("thr-model-2")
    assert time.time() - t0 < 0.5


def test_log_lines_from_many_threads_stay_whole(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "log.jsonl")
    threads = [threading.Thread(target=lambda i=i: [llm._log("judge", "m", f"k{i}-{j}" * 4, False, None, 0.1)
                                                    for j in range(25)]) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200 and all(json.loads(x)["role"] == "judge" for x in lines)