"""InferDPT end-to-end pipeline: perturb → remote generate → local extract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.llm import LLMClient
from inferdpt.rantext import Perturber

# Remote generation instruction (Ins): continue the perturbed prefix.
GENERATION_INSTRUCTION = (
    'Your task is to extend the "Prefix Text" with a coherent continuation.'
)


@dataclass
class Result:
    doc: str        # raw private document (never leaves the client)
    doc_p: str      # perturbed document sent to the remote model
    gen_p: str      # remote model's perturbed generation
    output: str     # final aligned generation


class InferDPT:
    def __init__(
        self,
        perturber: Perturber,
        gen_client: LLMClient,
        ext_client: LLMClient,
        *,
        epsilon: float = 3.0,
    ) -> None:
        self.perturber = perturber
        self.gen_client = gen_client
        self.ext_client = ext_client
        self.epsilon = epsilon

    def run(self, doc: str, *, seed: int | None = None) -> Result:
        doc_p = self.perturber.perturb(doc, self.epsilon, seed=seed)
        gen_p = self.gen_client.generate(
            f"{GENERATION_INSTRUCTION}\n\n- Prefix Text: {doc_p}"
        )
        output = extract(doc, gen_p, self.ext_client)
        return Result(doc=doc, doc_p=doc_p, gen_p=gen_p, output=output)


def default(
    vocab_cache: str | Path = "data/vocab",
    *,
    model: str = "gemma 4 (E4B)",
    epsilon: float = 3.0,
) -> InferDPT:
    """Build an InferDPT with both roles on the same model (initial test setup)."""
    perturber = Perturber(VocabEmbeddings.load(vocab_cache))
    # The served Gemma is a thinking model; disable reasoning so it emits content directly.
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    gen_client = LLMClient(model, temperature=0.7, max_tokens=256, extra_body=no_think)
    ext_client = LLMClient(model, temperature=0.3, max_tokens=256, extra_body=no_think)
    return InferDPT(perturber, gen_client, ext_client, epsilon=epsilon)
