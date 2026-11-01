from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from src import llm


class Out(BaseModel):
    verdict: str


class ApiErr(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def ok(text='{"verdict": "ok"}'):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3))


@pytest.fixture
def calls(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setitem(llm.MODELS, "reviewer", "fake-model")
    seen = []

    def fake(model, temperature, system, prompt, schema):
        seen.append(prompt)
        return ok()

    monkeypatch.setattr(llm, "_client_call", fake)
    return seen


def cached_files(tmp_path):
    return list((tmp_path / "cache").rglob("*.json"))


def test_identical_second_call_is_served_from_cache(calls):
    a = llm.call_llm("reviewer", "p", schema=Out)
    b = llm.call_llm("reviewer", "p", schema=Out)
    assert a == b == Out(verdict="ok")
    assert len(calls) == 1


def test_different_sample_is_a_new_call(calls):
    llm.call_llm("reviewer", "p", schema=Out, sample=0)
    llm.call_llm("reviewer", "p", schema=Out, sample=1)
    assert len(calls) == 2


def test_unset_model_raises_a_clear_error(calls, monkeypatch):
    monkeypatch.setitem(llm.MODELS, "critic", None)
    with pytest.raises(ValueError, match="critic"):
        llm.call_llm("critic", "p")


def test_daily_quota_stops_and_caches_nothing(calls, monkeypatch, tmp_path):
    def boom(*a):
        raise ApiErr(429, "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    monkeypatch.setattr(llm, "_client_call", boom)
    with pytest.raises(llm.DailyQuotaExhausted):
        llm.call_llm("reviewer", "p", schema=Out)
    assert cached_files(tmp_path) == []


def test_per_minute_429_is_retried(calls, monkeypatch):
    state = {"n": 0}

    def flaky(*a):
        state["n"] += 1
        if state["n"] < 3:
            raise ApiErr(429, "rate limit exceeded, retry shortly")
        return ok()

    monkeypatch.setattr(llm, "_client_call", flaky)
    assert llm.call_llm("reviewer", "p", schema=Out) == Out(verdict="ok")
    assert state["n"] == 3


def test_invalid_json_is_not_cached(calls, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_client_call", lambda *a: ok("not json"))
    with pytest.raises(ValidationError):
        llm.call_llm("reviewer", "p", schema=Out)
    assert cached_files(tmp_path) == []


def test_json_inside_code_fences_is_parsed(calls, monkeypatch):
    monkeypatch.setattr(llm, "_client_call", lambda *a: ok('```json\n{"verdict": "fenced"}\n```'))
    assert llm.call_llm("reviewer", "p", schema=Out) == Out(verdict="fenced")
