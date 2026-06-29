"""Extraction module — the local/trusted side of InferDPT.

A single call to the local model X: continue the raw prefix (Doc) using the remote
model's perturbed generation (Gen_p) as writing material. X is the only component
that sees the raw Doc, so it must be a trusted endpoint. Prompt is the paper's
(Appendix A / Fig. 3).
"""

from __future__ import annotations

from inferdpt.llm import LLMClient

EXTRACTION_INSTRUCTION = (
    'Your task is to extend the "Prefix Text". Use the "Perturbed Generation" as your '
    "primary writing material for your extension. Extract coherent and consistent text "
    'from the "Perturbed Generation" and integrate them into your continuation. Ensure '
    'a seamless alignment with the context established by the "Prefix Text".'
)


def extract(doc: str, gen_p: str, client: LLMClient, **overrides: object) -> str:
    """Reconstruct the aligned continuation of `doc` from perturbed generation `gen_p`."""
    prompt = f"{EXTRACTION_INSTRUCTION}\n\n- Prefix Text: {doc}\n\n- Perturbed Generation: {gen_p}"
    return client.generate(prompt, **overrides)
