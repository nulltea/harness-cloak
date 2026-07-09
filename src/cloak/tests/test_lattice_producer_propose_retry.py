import httpx
import pytest
from openai import APITimeoutError, RateLimitError
from cloak.lattice_producer import propose as propose_module
from cloak.lattice_producer.propose import _create_with_retry, _response_content


def _rate_limit_error():
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x/v1/chat/completions"))
    return RateLimitError("rate limited", response=resp, body=None)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    # the retry helper sleeps a backoff between attempts; skip the real wait in tests
    monkeypatch.setattr(propose_module.time, "sleep", lambda *_: None)


# the real OpenAI response is an object with a `.choices` attribute (not a dict), which is what
# _has_choices/_response_content inspect -- so the doubles mirror that shape.
class _OkResponse:
    def __init__(self, timeout=None):
        self.choices = [1]
        self.ok = True
        self.timeout = timeout


class _NoChoicesResponse:
    choices = None


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
        return _OkResponse(timeout=kwargs.get("timeout"))


def test_retry_succeeds_after_transient_timeouts():
    client = _FlakyClient(fail_times=2)
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert resp.ok and client.calls == 3


def test_retry_reraises_after_exhausting_attempts():
    client = _FlakyClient(fail_times=5)
    with pytest.raises(APITimeoutError):
        _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert client.calls == 3


def test_retry_covers_rate_limit_errors():
    # a 429 from a rate-limited (free) endpoint is retryable, not fatal
    client = _FlakyClient(fail_times=1, exc=_rate_limit_error())
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert resp.ok and client.calls == 2


class _EmptyThenOkClient:
    def __init__(self, empty_times):
        self.calls = 0
        self.empty_times = empty_times
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.empty_times:
            return _NoChoicesResponse()
        return _OkResponse()


def test_retry_recovers_from_transient_empty_choices():
    # a free endpoint can return a no-choices payload (provider error) that does not raise;
    # it is retried like a transient failure
    client = _EmptyThenOkClient(empty_times=1)
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert resp.ok and client.calls == 2


def test_retry_degrades_to_empty_response_instead_of_crashing():
    # a persistently empty response must NOT crash the run: the empty response is returned so the
    # caller degrades to a no-candidates diagnostic
    client = _EmptyThenOkClient(empty_times=99)
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert isinstance(resp, _NoChoicesResponse) and client.calls == 3
    assert _response_content(resp) == "{}"
