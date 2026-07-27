"""Tests for LLMClient.refresh (cache-bypass recompute) and single_flight (per-model lock)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from cloak import llm
from cloak.llm import LLMClient


def _resp(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeTransport:
    """Stand-in for OpenAI client: exposes .chat.completions.create and tracks calls/overlap."""

    def __init__(self, reply="v", delay=0.0):
        self.chat = self
        self.completions = self
        self.reply = reply
        self.delay = delay
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def create(self, **params):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.calls.append(params)
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        reply = self.reply(params) if callable(self.reply) else self.reply
        return _resp(reply)


def _make_client(monkeypatch, tmp_path, *, model="m", base_url="http://fake", single_flight=False, transport=None):
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path))
    client = LLMClient(model, base_url=base_url, temperature=0.0, single_flight=single_flight)
    client._client = transport or FakeTransport()
    return client


def _cache_files(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".json"))


def test_refresh_bypasses_and_overwrites(monkeypatch, tmp_path):
    msgs = [{"role": "user", "content": "hi"}]
    c = _make_client(monkeypatch, tmp_path, transport=FakeTransport(reply="first"))
    assert c.chat(msgs) == "first"
    files_before = _cache_files(tmp_path)
    assert len(files_before) == 1

    # cached: normal call does not recompute
    c._client.reply = "second"
    assert c.chat(msgs) == "first"
    assert len(c._client.calls) == 1

    # refresh: bypass cache, recompute, overwrite same file
    assert c.chat(msgs, refresh=True) == "second"
    assert len(c._client.calls) == 2
    assert _cache_files(tmp_path) == files_before  # same path

    # subsequent normal call reads the refreshed value
    assert c.chat(msgs) == "second"
    assert len(c._client.calls) == 2


def test_cache_write_uses_atomic_replace(monkeypatch, tmp_path):
    calls = []
    real_replace = llm.os.replace

    def replace_spy(src, dst):
        calls.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(llm.os, "replace", replace_spy)
    c = _make_client(monkeypatch, tmp_path, transport=FakeTransport(reply="atomic"))

    assert c.chat([{"role": "user", "content": "write"}]) == "atomic"

    assert len(calls) == 1
    src, dst = calls[0]
    assert src != dst
    assert llm.os.path.dirname(llm.os.fspath(src)) == llm.os.path.dirname(llm.os.fspath(dst))
    assert _cache_files(tmp_path) == [llm.os.path.basename(llm.os.fspath(dst))]


def test_empty_provider_completion_is_not_cached_and_records_no_choices(monkeypatch, tmp_path):
    class EmptyChoicesTransport(FakeTransport):
        def create(self, **params):
            self.calls.append(params)
            return SimpleNamespace(choices=[])

    c = _make_client(monkeypatch, tmp_path, transport=EmptyChoicesTransport())

    assert c.chat([{"role": "user", "content": "write"}]) == ""
    assert c.last_completion_state == {"outcome": "no_choices"}
    assert _cache_files(tmp_path) == []


def test_single_flight_double_checks_cache_for_cold_key(monkeypatch, tmp_path):
    transport = FakeTransport(reply="once", delay=0.1)
    c = _make_client(monkeypatch, tmp_path, single_flight=True, transport=transport)
    msgs = [{"role": "user", "content": "same cold key"}]
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait()
            results.append(c.chat(msgs))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    assert results == ["once", "once"]
    assert len(transport.calls) == 1


def test_corrupt_cache_file_is_treated_as_miss(monkeypatch, tmp_path):
    msgs = [{"role": "user", "content": "corrupt"}]
    c = _make_client(monkeypatch, tmp_path, transport=FakeTransport(reply="fresh"))

    assert c.chat(msgs) == "fresh"
    cache_path = next(tmp_path.glob("*.json"))
    cache_path.write_text('{"content": ')

    c._client.reply = "recovered"
    assert c.chat(msgs) == "recovered"
    assert len(c._client.calls) == 2
    assert json.loads(cache_path.read_text())["content"] == "recovered"


def test_refresh_not_in_params(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.chat([{"role": "user", "content": "x"}], refresh=True)
    assert c._client.calls, "transport not called"
    for params in c._client.calls:
        assert "refresh" not in params


def test_single_flight_serializes(monkeypatch, tmp_path):
    t = FakeTransport(delay=0.05)
    c = _make_client(monkeypatch, tmp_path, single_flight=True, transport=t)

    def worker(i):
        c.chat([{"role": "user", "content": f"q{i}"}])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.max_active == 1


def test_without_single_flight_overlaps(monkeypatch, tmp_path):
    t = FakeTransport(delay=0.05)
    c = _make_client(monkeypatch, tmp_path, single_flight=False, transport=t)

    def worker(i):
        c.chat([{"role": "user", "content": f"q{i}"}])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.max_active > 1


def test_lock_shared_across_instances_same_key(monkeypatch, tmp_path):
    shared = FakeTransport(delay=0.05)
    c1 = _make_client(monkeypatch, tmp_path, model="same", single_flight=True, transport=shared)
    c2 = _make_client(monkeypatch, tmp_path, model="same", single_flight=True, transport=shared)

    def w(c, i):
        c.chat([{"role": "user", "content": f"q{i}"}])

    threads = [threading.Thread(target=w, args=(c1, 0)), threading.Thread(target=w, args=(c2, 1))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert shared.max_active == 1


def test_different_model_different_lock(monkeypatch, tmp_path):
    shared = FakeTransport(delay=0.05)
    c1 = _make_client(monkeypatch, tmp_path, model="A", single_flight=True, transport=shared)
    c2 = _make_client(monkeypatch, tmp_path, model="B", single_flight=True, transport=shared)

    def w(c, i):
        c.chat([{"role": "user", "content": f"q{i}"}])

    threads = [threading.Thread(target=w, args=(c1, 0)), threading.Thread(target=w, args=(c2, 1))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert shared.max_active > 1


def test_cache_hit_does_not_acquire_lock(monkeypatch, tmp_path):
    msgs = [{"role": "user", "content": "cached"}]
    c = _make_client(monkeypatch, tmp_path, single_flight=True, transport=FakeTransport(reply="hit"))
    assert c.chat(msgs) == "hit"  # populate cache

    # Hold the per-(base_url, model) lock; a cache-hit read must not try to acquire it.
    lock = LLMClient._lock_for(c.base_url, c.model)
    result: list[str] = []
    lock.acquire()
    try:
        th = threading.Thread(target=lambda: result.append(c.chat(msgs)))
        th.start()
        th.join(timeout=2.0)
        assert not th.is_alive(), "cache-hit read blocked on the single-flight lock"
    finally:
        lock.release()
    assert result == ["hit"]


def test_request_scope_overlaps_distinct_prompts(monkeypatch, tmp_path):
    t = FakeTransport(delay=0.05)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path))
    c = LLMClient(
        "m", base_url="http://fake", temperature=0.0,
        single_flight=True, single_flight_scope="request",
    )
    c._client = t

    def worker(i):
        c.chat([{"role": "user", "content": f"q{i}"}])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.max_active > 1
    assert len(t.calls) == 4


def test_request_scope_still_dedupes_identical_prompts(monkeypatch, tmp_path):
    t = FakeTransport(delay=0.05)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path))
    c = LLMClient(
        "m", base_url="http://fake", temperature=0.0,
        single_flight=True, single_flight_scope="request",
    )
    c._client = t

    def worker():
        c.chat([{"role": "user", "content": "same"}])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.max_active == 1
    assert len(t.calls) == 1


def test_request_scope_rejects_unknown_value():
    with pytest.raises(ValueError, match="single_flight_scope"):
        LLMClient("m", base_url="http://fake", single_flight_scope="bogus")


def test_cache_path_uses_cloak_env_name(monkeypatch, tmp_path):
    """The disk-cache env var is CLOAK_LLM_CACHE (the inferdpt-era name is retired)."""
    from cloak.llm import _cache_path

    monkeypatch.delenv("CLOAK_LLM_CACHE", raising=False)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path))

    path = _cache_path(
        "model",
        [{"role": "user", "content": "hello"}],
        {"temperature": 0.0},
        "http://example.test/v1",
    )

    assert path is not None
    assert path.startswith(str(tmp_path))
    assert path.endswith(".json")
