import pytest
from openai import APITimeoutError
from cloak.lattice_producer.propose import _create_with_retry


class _FlakyClient:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise APITimeoutError(request=None)
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
