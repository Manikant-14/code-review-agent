import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src import llm


class Out(BaseModel):
    verdict: str


class ApiErr(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.status_code = code


def test_split_uses_first_colon_and_rejects_bad_ids():
    assert llm._split("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert llm._split("gemini:gemma-4-31b-it") == ("gemini", "gemma-4-31b-it")
    for bad in ("gemma-4-31b-it", "nope:model", ""):
        with pytest.raises(ValueError):
            llm._split(bad)


def test_client_call_routes_by_provider(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_gemini_call", lambda *a: seen.append(("gemini", a[0])))
    monkeypatch.setattr(llm, "_openai_call", lambda *a: seen.append(("openai", a[0], a[1])))
    llm._client_call("gemini:gemma-4-31b-it", 0.0, None, "p", None)
    llm._client_call("groq:openai/gpt-oss-120b", 0.0, None, "p", None)
    assert seen == [("gemini", "gemma-4-31b-it"), ("openai", "groq", "openai/gpt-oss-120b")]


def test_openai_adapter_end_to_end_through_call_llm(monkeypatch, tmp_path):
    captured = {}

    def create(**kw):
        captured.update(kw)
        msg = SimpleNamespace(content='{"verdict": "ok"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4))

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_get_openai_client", lambda provider: fake)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "log.jsonl")
    monkeypatch.setitem(llm.MODELS, "reviewer", "groq:openai/gpt-oss-120b")

    assert llm.call_llm("reviewer", "hello", system="be terse", schema=Out) == Out(verdict="ok")
    assert captured["model"] == "openai/gpt-oss-120b"  # provider prefix stripped
    assert captured["response_format"] == {"type": "json_object"}
    assert "be terse" in captured["messages"][0]["content"]
    assert "verdict" in captured["messages"][0]["content"]  # schema is in the system text
    row = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert (row["prompt_tokens"], row["output_tokens"]) == (11, 4)


def test_groq_error_messages_are_classified():
    tpd = ApiErr(429, "Rate limit reached ... on tokens per day (TPD): Limit 200000")
    rpd = ApiErr(429, "Rate limit reached ... on requests per day (RPD): Limit 1000")
    tpm = ApiErr(429, "Rate limit reached ... on tokens per minute (TPM): Limit 8000. Try again in 7s")
    big = ApiErr(413, "Request too large for model ... on tokens per minute (TPM): Limit 8000, Requested 9500")
    assert [llm._classify(e) for e in (tpd, rpd, tpm, big)] == ["daily", "daily", "retry", "fatal"]
