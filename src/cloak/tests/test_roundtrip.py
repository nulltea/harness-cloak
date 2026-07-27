"""Round-trip reward wiring — offline (remote client construction only)."""
import cloak.reward.roundtrip as rt


def test_remote_gen_client_uses_single_flight(monkeypatch):
    seen = {}

    class FakeLLMClient:
        def __init__(self, model, **kwargs):
            seen["model"] = model
            seen["kwargs"] = kwargs

        def generate(self, prompt, **kwargs):
            return "unused"

    import cloak.llm as llm
    monkeypatch.setenv("CLOAK_LLM_CACHE", "test-cache")
    monkeypatch.setattr(llm, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(rt, "_client", None)

    try:
        rt._remote()
    finally:
        monkeypatch.setattr(rt, "_client", None)

    assert seen["model"] == rt.RT_MODEL
    assert seen["kwargs"]["single_flight"] is True
