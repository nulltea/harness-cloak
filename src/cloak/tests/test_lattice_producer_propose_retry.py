import httpx
import pytest
from openai import APITimeoutError, RateLimitError
from cloak.lattice_producer import propose as propose_module
from cloak.lattice_producer.propose import _create_with_retry


def _rate_limit_error():
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x/v1/chat/completions"))
    return RateLimitError("rate limited", response=resp, body=None)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    # the retry helper sleeps a backoff between attempts; skip the real wait in tests
    monkeypatch.setattr(propose_module.time, "sleep", lambda *_: None)


class _FlakyClient:
    def __init__(self, fail_times, exc=None):
        self.calls = 0
        self.fail_times = fail_times
        self._exc = exc or APITimeoutError(request=None)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self._exc
        return {"ok": True, "timeout": kwargs.get("timeout")}


def test_retry_succeeds_after_transient_timeouts():
    client = _FlakyClient(fail_times=2)
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert resp["ok"] and client.calls == 3


def test_retry_reraises_after_exhausting_attempts():
    client = _FlakyClient(fail_times=5)
    with pytest.raises(APITimeoutError):
        _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert client.calls == 3


def test_retry_covers_rate_limit_errors():
    # a 429 from a rate-limited (free) endpoint is retryable, not fatal
    client = _FlakyClient(fail_times=1, exc=_rate_limit_error())
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert resp["ok"] and client.calls == 2
