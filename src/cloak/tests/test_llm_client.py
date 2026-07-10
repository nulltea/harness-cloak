"""Tests for LLMClient.refresh (cache-bypass recompute) and single_flight (per-model lock)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

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
