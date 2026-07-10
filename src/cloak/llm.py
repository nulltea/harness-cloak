"""OpenAI-compatible chat client for Cloak remote/model-server calls."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading

from openai import OpenAI

DEFAULT_BASE_URL = "https://ai.tail59ea6b.ts.net/v1"

Message = dict[str, str]

# Guards the class-level single-flight lock registry (LLMClient._locks).
_registry_lock = threading.Lock()


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


def _read_cache(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)["content"]
    except json.JSONDecodeError:
        return None


def _write_cache(path: str, content: str, model: str) -> None:
    cache_dir = os.path.dirname(path)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=cache_dir, delete=False) as f:
            tmp_path = f.name
            json.dump({"content": content, "model": model}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


class LLMClient:
    """A chat model behind an OpenAI-compatible API.

    Defaults passed at construction, such as temperature and max_tokens, apply to every
    call and can be overridden per call.
    """

    # Per-(base_url, model) locks shared across all instances; access via _registry_lock.
    _locks: dict[tuple[str, str], threading.Lock] = {}

    @classmethod
    def _lock_for(cls, base_url: str, model: str) -> threading.Lock:
        key = (base_url, model)
        with _registry_lock:
            lock = cls._locks.get(key)
            if lock is None:
                lock = cls._locks[key] = threading.Lock()
        return lock

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        single_flight: bool = False,
        **defaults: object,
    ) -> None:
        self.model = model
        self._defaults = defaults
        self.single_flight = single_flight
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
        )

    def chat(self, messages: list[Message], *, refresh: bool = False, **overrides: object) -> str:
        """Return the assistant reply text for chat messages, cached when configured.

        With refresh=True, skip the cache read, recompute, and overwrite the cache file.
        """
        params = {**self._defaults, **overrides}
        path = _cache_path(self.model, messages, params, self.base_url)
        if path and not refresh:
            cached = _read_cache(path)
            if cached is not None:
                return cached
        if self.single_flight:
            with self._lock_for(self.base_url, self.model):
                if path and not refresh:
                    cached = _read_cache(path)
                    if cached is not None:
                        return cached
                return self._compute(messages, params, path)
        return self._compute(messages, params, path)

    def _compute(self, messages: list[Message], params: dict, path: str | None) -> str:
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        content = resp.choices[0].message.content or ""
        if path:
            _write_cache(path, content, self.model)
        return content

    def generate(
        self, prompt: str, *, system: str | None = None, refresh: bool = False, **overrides: object
    ) -> str:
        """Convenience wrapper for a single-turn prompt with an optional system message."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, refresh=refresh, **overrides)


if __name__ == "__main__":
    client = LLMClient("gemma 4 (E4B)", temperature=0.0, max_tokens=64)
    reply = client.generate("Reply with exactly the word: ready")
    print("reply:", repr(reply))
    assert reply.strip(), "empty response from endpoint"
    print("OK")
