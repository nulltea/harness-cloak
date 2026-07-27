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
    # max_tokens is a generation SAFETY bound, not a semantic input: a call that terminates before
    # the cap yields identical content with or without it. Excluding it from the key means adding a
    # cap to stop a runaway generation does NOT invalidate every prior cached verdict. (A call that
    # genuinely truncates at the cap is a degenerate output we would not want to reuse anyway.)
    key_params = {k: v for k, v in params.items() if k != "max_tokens"}
    blob = json.dumps(
        {
            "model": model,
            "base_url": base_url,
            "messages": messages,
            "params": key_params,
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


def _record_reasoning(cache_path: str | None, messages: list[Message], content: str,
                      reasoning: str | None, finish_reason, model: str) -> None:
    """Dev-only sidecar of the model's reasoning trace, for prompt A/B tweaking. OFF unless
    $CLOAK_LLM_REASONING_DIR is set. Keyed by the cache hash so reasoning ties to prompt+output;
    never read by the pipeline and never affects the cache key or the chat return value."""
    out_dir = os.getenv("CLOAK_LLM_REASONING_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    key = (os.path.splitext(os.path.basename(cache_path))[0] if cache_path
           else hashlib.sha256((content or "").encode()).hexdigest())
    path = os.path.join(out_dir, key + ".json")
    try:
        with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False) as f:
            tmp = f.name
            json.dump({"model": model, "finish_reason": finish_reason, "reasoning": reasoning,
                       "content": content, "messages": messages}, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


class LLMClient:
    """A chat model behind an OpenAI-compatible API.

    Defaults passed at construction, such as temperature and max_tokens, apply to every
    call and can be overridden per call.
    """

    # Per-(base_url, model) locks shared across all instances; access via _registry_lock.
    _locks: dict[tuple[str, str], threading.Lock] = {}
    # Striped locks for request-scoped single flight: identical cache paths always share
    # a stripe (stampede protection), distinct requests almost always run concurrently.
    _request_locks: tuple[threading.Lock, ...] = tuple(
        threading.Lock() for _ in range(64)
    )

    @classmethod
    def _lock_for(cls, base_url: str, model: str) -> threading.Lock:
        key = (base_url, model)
        with _registry_lock:
            lock = cls._locks.get(key)
            if lock is None:
                lock = cls._locks[key] = threading.Lock()
        return lock

    @classmethod
    def _request_lock_for(cls, path: str) -> threading.Lock:
        return cls._request_locks[hash(path) % len(cls._request_locks)]

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        single_flight: bool = False,
        single_flight_scope: str = "model",
        max_retries: int | None = None,
        **defaults: object,
    ) -> None:
        if single_flight_scope not in {"model", "request"}:
            raise ValueError("single_flight_scope must be 'model' or 'request'")
        self.model = model
        self._defaults = defaults
        self.single_flight = single_flight
        self.single_flight_scope = single_flight_scope
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        # The OpenAI SDK already retries 408/409/429/>=500 with exponential backoff and honors
        # Retry-After; its default cap of 2 is too low for throttled free tiers (OpenRouter
        # :free 429s in bursts), so lift it. CLOAK_LLM_MAX_RETRIES overrides for a hard run.
        if max_retries is None:
            max_retries = int(os.getenv("CLOAK_LLM_MAX_RETRIES", "8"))
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
            max_retries=max_retries,
        )
        # A deliberately small diagnostic for callers that must distinguish an
        # absent provider completion from malformed model text.  Never retain
        # prompt or response content here.
        self.last_completion_state: dict[str, str | None] = {"outcome": "not_called"}

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
            # Request scope serializes only identical requests (same cache path), so
            # distinct prompts use the server's parallel slots; without a cache path
            # there is no stampede to prevent and the model-wide lock is kept.
            if self.single_flight_scope == "request" and path:
                lock = self._request_lock_for(path)
            else:
                lock = self._lock_for(self.base_url, self.model)
            with lock:
                if path and not refresh:
                    cached = _read_cache(path)
                    if cached is not None:
                        return cached
                return self._compute(messages, params, path)
        return self._compute(messages, params, path)

    def _compute(self, messages: list[Message], params: dict, path: str | None) -> str:
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        # Throttled/free endpoints (OpenRouter :free) can return HTTP 200 with no choices — an
        # error payload the SDK's max_retries does not catch. Degrade to "" WITHOUT caching, so
        # the empty is treated as a miss and re-tried on the next run instead of crashing on
        # resp.choices[0] or poisoning the cache.
        if not resp.choices:
            self.last_completion_state = {"outcome": "no_choices"}
            return ""
        choice = resp.choices[0]
        content = choice.message.content or ""
        reasoning = (getattr(choice.message, "reasoning", None)
                     or getattr(choice.message, "reasoning_content", None)
                     or (getattr(choice.message, "model_extra", None) or {}).get("reasoning"))
        _record_reasoning(path, messages, content, reasoning,
                          getattr(choice, "finish_reason", None), self.model)
        if not content.strip():
            self.last_completion_state = {
                "outcome": "empty_content",
                "finish_reason": getattr(choice, "finish_reason", None),
            }
            return content
        self.last_completion_state = {
            "outcome": "content",
            "finish_reason": getattr(choice, "finish_reason", None),
        }
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
