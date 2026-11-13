"""LLM client: disk cache, per-model rate limiting, retries and a JSONL call log.

Every call is keyed on (model, temperature, system, prompt, schema, sample). A cache hit costs
nothing, so reruns, ablations and passes interrupted by a quota stop all resume for free.
Cache and log live under data/ (git-ignored).
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Type

from pydantic import BaseModel

from src.config import DEFAULT_RPM, MODELS, PROVIDERS, RPM_LIMITS, TEMPERATURES

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "llm_cache"
LOG_PATH = ROOT / "data" / "llm_log.jsonl"
MAX_RETRIES = 5


class DailyQuotaExhausted(RuntimeError):
    """The daily request quota is used up. Rerun later: cached calls are free."""


_client = None
_last_call: dict[str, float] = {}


def _get_client():
    global _client
    if _client is None:
        from google import genai  # lazy: tests never need the SDK

        _client = genai.Client()  # reads GEMINI_API_KEY from the environment
    return _client


def _is_gemma(model: str) -> bool:
    return model.lower().startswith(("gemma", "models/gemma"))


def _split(model: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'). Splits on the first colon."""
    provider, sep, name = model.partition(":")
    if not sep or provider not in PROVIDERS:
        raise ValueError(f"model id '{model}' must be 'provider:model' with provider in {sorted(PROVIDERS)}")
    return provider, name


def _client_call(model, temperature, system, prompt, schema):
    """Route to the provider named in the model id. Every adapter returns an object with
    .text and .usage_metadata(prompt_token_count, candidates_token_count)."""
    provider, name = _split(model)
    if PROVIDERS[provider]["kind"] == "gemini":
        return _gemini_call(name, temperature, system, prompt, schema)
    return _openai_call(provider, name, temperature, system, prompt, schema)


_openai_clients: dict = {}


class ApiError(Exception):
    """HTTP error from an OpenAI-compatible API; carries status_code so _classify can read it."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _HttpCompletions:
    """Minimal chat-completions client over plain httpx. Used instead of the openai SDK because
    the SDK pulls in a compiled dependency (jiter) that some Windows Application Control policies
    block. Exposes .create(**payload) with the same response shape the SDK has."""

    def __init__(self, base_url: str, key: str):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                        "User-Agent": "code-review-agent/0.1"}

    def create(self, **payload):
        import httpx

        r = httpx.post(self.url, headers=self.headers, json=payload, timeout=120)
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.text)
        data = r.json()
        u = data.get("usage") or {}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=data["choices"][0]["message"].get("content")))],
            usage=SimpleNamespace(prompt_tokens=u.get("prompt_tokens"),
                                  completion_tokens=u.get("completion_tokens")))


def _get_openai_client(provider: str):
    if provider not in _openai_clients:
        from dotenv import load_dotenv  # lazy: tests never need this

        load_dotenv()  # fills missing env vars from .env, never overrides real ones
        cfg = PROVIDERS[provider]
        key_env = cfg.get("key_env")
        key = os.environ.get(key_env) if key_env else "none"
        if not key:
            raise RuntimeError(f"{key_env} is not set: add it to .env")
        # No SDK retries: our own throttle/retry loop is the only retry layer, so call counts
        # and waits stay predictable against free-tier quotas.
        _openai_clients[provider] = SimpleNamespace(
            chat=SimpleNamespace(completions=_HttpCompletions(cfg["base_url"], key)))
    return _openai_clients[provider]


def _openai_call(provider, name, temperature, system, prompt, schema):
    sys_text = system or ""
    kwargs = {}
    if schema is not None:
        # json_object mode works on every OpenAI-compatible provider; the schema goes in the
        # system text, and call_llm validates the reply with pydantic afterwards.
        sys_text += ("\n\nReply with one JSON object matching this JSON schema, nothing else:\n"
                     + json.dumps(schema.model_json_schema()))
        kwargs["response_format"] = {"type": "json_object"}
    messages = ([{"role": "system", "content": sys_text.strip()}] if sys_text.strip() else [])
    messages.append({"role": "user", "content": prompt})
    resp = _get_openai_client(provider).chat.completions.create(
        model=name, messages=messages, temperature=temperature, **kwargs)
    u = getattr(resp, "usage", None)
    return SimpleNamespace(
        text=resp.choices[0].message.content or "",
        usage_metadata=SimpleNamespace(prompt_token_count=getattr(u, "prompt_tokens", None),
                                       candidates_token_count=getattr(u, "completion_tokens", None)))


def _gemini_call(model, temperature, system, prompt, schema):
    from google.genai import types

    cfg = {"temperature": temperature}
    if _is_gemma(model):
        # Gemma endpoints may reject system instructions and JSON mode: fold the system text
        # into the prompt and parse JSON from the reply text instead.
        if system:
            prompt = f"{system}\n\n{prompt}"
    else:
        if system:
            cfg["system_instruction"] = system
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
    return _get_client().models.generate_content(
        model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
    )


def _throttle(model: str) -> None:
    gap = 60.0 / RPM_LIMITS.get(model, DEFAULT_RPM) * 1.05
    wait = _last_call.get(model, 0.0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[model] = time.time()


def _classify(e: Exception) -> str:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    msg = str(e).lower().replace(" ", "")
    if "requesttoolarge" in msg:  # prompt alone exceeds the per-minute token limit: retrying is futile
        return "fatal"
    if code == 429:
        return "daily" if ("perday" in msg or "daily" in msg) else "retry"
    if code in (500, 502, 503, 504):
        return "retry"
    return "fatal"


def _extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _key(model, temperature, system, prompt, schema, sample) -> str:
    blob = json.dumps(
        {"model": model, "temperature": temperature, "system": system, "prompt": prompt,
         "schema": schema.__name__ if schema else None, "sample": sample},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _log(role, model, key, cached, usage, latency) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), "role": role, "model": model, "key": key[:12],
           "cached": cached, "latency_s": round(latency, 3), **(usage or {})}
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def _generate(model, temperature, system, prompt, schema):
    for attempt in range(MAX_RETRIES):
        _throttle(model)
        t0 = time.time()
        try:
            resp = _client_call(model, temperature, system, prompt, schema)
        except Exception as e:  # noqa: BLE001
            kind = _classify(e)
            if kind == "daily":
                raise DailyQuotaExhausted(str(e)[:300]) from e
            if kind == "retry" and attempt < MAX_RETRIES - 1:
                time.sleep(min(60, 5 * 2**attempt))
                continue
            raise
        um = getattr(resp, "usage_metadata", None)
        usage = {"prompt_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None)}
        return resp.text or "", usage, time.time() - t0
    raise RuntimeError("unreachable")


def call_llm(role: str, prompt: str, *, system: Optional[str] = None,
             schema: Optional[Type[BaseModel]] = None, sample: int = 0, use_cache: bool = True):
    """Call the model assigned to `role`. Returns a parsed `schema` object, or text if no schema.

    `sample` distinguishes repeated runs of the same prompt (different cache entries).
    """
    model = MODELS.get(role)
    if not model:
        raise ValueError(f"no model chosen for role '{role}': set MODELS['{role}'] in src/config.py")
    temperature = TEMPERATURES[role]
    key = _key(model, temperature, system, prompt, schema, sample)
    path = CACHE_DIR / key[:2] / f"{key}.json"

    if use_cache and path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        _log(role, model, key, True, rec.get("usage"), 0.0)
        text = rec["text"]
    else:
        text, usage, latency = _generate(model, temperature, system, prompt, schema)
        if schema is not None:
            schema.model_validate_json(_extract_json(text))  # raises before anything is cached
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": role, "model": model, "text": text, "usage": usage}),
                        encoding="utf-8", newline="\n")
        _log(role, model, key, False, usage, latency)

    return schema.model_validate_json(_extract_json(text)) if schema is not None else text
