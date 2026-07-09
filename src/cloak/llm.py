"""OpenAI-compatible chat client for Cloak remote/model-server calls."""

from __future__ import annotations

import hashlib
import json
import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://ai.tail59ea6b.ts.net/v1"

Message = dict[str, str]


def _cache_path(
    model: str,
    messages: list[Message],
    params: dict,
    base_url: str | None = None,
) -> str | None:
    """Content-addressed cache file under $CLOAK_LLM_CACHE, or None when disabled."""
    cache_dir = os.getenv("CLOAK_LLM_CACHE")
    if not cache_dir:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    blob = json.dumps(
        {
            "model": model,
            "base_url": base_url,
            "messages": messages,
            "params": params,
        },
        sort_keys=True,
        default=str,
    )
    return os.path.join(cache_dir, hashlib.sha256(blob.encode()).hexdigest() + ".json")


class LLMClient:
    """A chat model behind an OpenAI-compatible API.

    Defaults passed at construction, such as temperature and max_tokens, apply to every
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
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
        )

    def chat(self, messages: list[Message], **overrides: object) -> str:
        """Return the assistant reply text for chat messages, cached when configured."""
        params = {**self._defaults, **overrides}
        path = _cache_path(self.model, messages, params, self.base_url)
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)["content"]
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        content = resp.choices[0].message.content or ""
        if path:
            with open(path, "w") as f:
                json.dump({"content": content, "model": self.model}, f)
        return content

    def generate(self, prompt: str, *, system: str | None = None, **overrides: object) -> str:
        """Convenience wrapper for a single-turn prompt with an optional system message."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)


if __name__ == "__main__":
    client = LLMClient("gemma 4 (E4B)", temperature=0.0, max_tokens=64)
    reply = client.generate("Reply with exactly the word: ready")
    print("reply:", repr(reply))
    assert reply.strip(), "empty response from endpoint"
    print("OK")
