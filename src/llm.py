"""LLM client: disk cache, per-model rate limiting, retries and a JSONL call log.

Every call is keyed on (model, temperature, system, prompt, schema, sample). A cache hit costs
nothing, so reruns, ablations and passes interrupted by a quota stop all resume for free.
Cache and log live under data/ (git-ignored).
"""
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional, Type

from pydantic import BaseModel

from src.config import DEFAULT_RPM, MODELS, RPM_LIMITS, TEMPERATURES

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


def _client_call(model, temperature, system, prompt, schema):
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
