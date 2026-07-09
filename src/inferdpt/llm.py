"""Compatibility wrapper for the Cloak OpenAI-compatible chat client."""

from __future__ import annotations

from cloak.llm import DEFAULT_BASE_URL, LLMClient, Message, _cache_path


if __name__ == "__main__":
    client = LLMClient("gemma 4 (E4B)", temperature=0.0, max_tokens=64)
    reply = client.generate("Reply with exactly the word: ready")
    print("reply:", repr(reply))
    assert reply.strip(), "empty response from endpoint"
    print("OK")
