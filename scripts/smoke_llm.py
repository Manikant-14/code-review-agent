"""One real call to a provider:model id, to prove keys, routing, JSON mode and usage logging.

Usage (repo root, venv active):  python -m scripts.smoke_llm groq:openai/gpt-oss-120b
Bypasses the cache so it always hits the API. Costs one request.
"""
import json
import sys
import time

from pydantic import BaseModel

from src import config, llm


class Ping(BaseModel):
    answer: int
    note: str


def main(model: str) -> None:
    config.MODELS["reviewer"] = model  # llm.py shares this dict
    t0 = time.time()
    out = llm.call_llm(
        "reviewer",
        "What is 17 * 3? Put the number in 'answer' and one short sentence in 'note'.",
        system="You are terse.", schema=Ping, use_cache=False)
    print(f"model   : {model}")
    print(f"parsed  : {out}")
    print(f"elapsed : {time.time() - t0:.1f}s")
    last = json.loads(llm.LOG_PATH.read_text(encoding="utf-8").splitlines()[-1])
    print(f"usage   : prompt={last.get('prompt_tokens')} output={last.get('output_tokens')}")
    assert out.answer == 51, "wrong answer: model reachable but not following the prompt"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.smoke_llm provider:model")
    main(sys.argv[1])
