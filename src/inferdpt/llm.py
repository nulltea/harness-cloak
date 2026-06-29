"""OpenAI-compatible chat client.

One thin wrapper over the official `openai` SDK, pointed at any OpenAI-compatible
endpoint. InferDPT uses two instances of this — one for the remote black-box
*generation* model (Y) and one for the local/trusted *extraction* model (X) —
but they share this single interface.
"""

from __future__ import annotations

import hashlib
import json
import os

from openai import OpenAI

# ts-llm-proxy (OpenAI-compatible). Override per call or via OPENAI_BASE_URL.
DEFAULT_BASE_URL = "https://ai.tail59ea6b.ts.net/v1"

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


def _cache_path(model: str, messages: list[Message], params: dict) -> str | None:
    """Content-addressed cache file under $INFERDPT_LLM_CACHE, or None when disabled.
    Caching a fixed (model, messages, params) freezes one sample → reproducible sweeps and
    no recapture of identical prompts across runs."""
    cache_dir = os.getenv("INFERDPT_LLM_CACHE")
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    blob = json.dumps({"model": model, "messages": messages, "params": params},
                      sort_keys=True, default=str)
    return os.path.join(cache_dir, hashlib.sha256(blob.encode()).hexdigest() + ".json")


class LLMClient:
    """A chat model behind an OpenAI-compatible API.

    Defaults passed at construction (e.g. temperature, max_tokens) apply to every
    call and can be overridden per call.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **defaults: object,
    ) -> None:
        self.model = model
        self._defaults = defaults
        # The proxy needs no key for local models; SDK still requires a non-empty one.
        self._client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
        )

    def chat(self, messages: list[Message], **overrides: object) -> str:
        """Return the assistant's reply text for a list of chat messages (disk-cached if
        $INFERDPT_LLM_CACHE is set)."""
        params = {**self._defaults, **overrides}
        path = _cache_path(self.model, messages, params)
        if path and os.path.exists(path):
            return json.loads(open(path).read())["content"]
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        content = resp.choices[0].message.content or ""
        if path:
            open(path, "w").write(json.dumps({"content": content, "model": self.model}))
        return content

    def generate(self, prompt: str, *, system: str | None = None, **overrides: object) -> str:
        """Convenience for a single-turn prompt (optional system message)."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)


if __name__ == "__main__":
    # Live smoke test against the default endpoint.
    client = LLMClient("gemma 4 (E4B)", temperature=0.0, max_tokens=64)
    reply = client.generate("Reply with exactly the word: ready")
    print("reply:", repr(reply))
    assert reply.strip(), "empty response from endpoint"
    print("OK")
