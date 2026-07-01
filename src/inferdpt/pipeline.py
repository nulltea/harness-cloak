"""InferDPT end-to-end pipeline: perturb → remote generate → local extract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.llm import LLMClient
from inferdpt.rantext import Perturber

# Remote generation prompt (Ins): continue the perturbed prefix. Wording matches the
# reference repo (mengtong0110/InferDPT main.py) — the instruction brackets the prefix,
# with "Provide only your Continuation" *after* it.
def gen_prompt(prefix: str) -> str:
    return (
        "Your task is to extend Prefix Text.\n"
        f"- Prefix Text:{prefix}\n\n"
        " Provide only your Continuation.\n"
        "- Continuation:"
    )


def pmap(fn, items, workers: int = 8):
    """Map fn over items concurrently, order-preserving. For remote LLM calls only — the
    proxy batches concurrent requests (~7x at 8 workers). Never wrap GPU probes (not thread-safe)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def load_corpus(path: str | Path) -> tuple[list[str], list[str] | None]:
    """Read a corpus → (prefixes, golds). `.jsonl` with {"prefix","gold"} per line →
    both (gold = paper-faithful MAUVE reference); plain `.txt` (one prefix per line) →
    golds is None (caller falls back to control generations)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        return [r["prefix"] for r in rows], [r["gold"] for r in rows]
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()], None


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
        gen_p = self.gen_client.generate(gen_prompt(doc_p))
        output = extract(doc, gen_p, self.ext_client)
        return Result(doc=doc, doc_p=doc_p, gen_p=gen_p, output=output)


def default(
    vocab_cache: str | Path = "data/vocab",
    *,
    gen_model: str = "Qwen3.6-35B-A3B",
    ext_model: str = "gemma 4 (E4B)",
    epsilon: float = 3.0,
) -> InferDPT:
    """Build an InferDPT with the remote generator Y and local extractor X on *different*
    models (gen=Qwen3.6-35B-A3B, ext=gemma-4-E4B) — distinct models reduce the shared-model
    bias in the Gen_p ablation probe."""
    perturber = Perturber(VocabEmbeddings.load(vocab_cache))
    # These are thinking models; disable reasoning so they emit content directly.
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    gen_client = LLMClient(gen_model, temperature=0.7, max_tokens=256, extra_body=no_think)
    ext_client = LLMClient(ext_model, temperature=0.3, max_tokens=256, extra_body=no_think)
    return InferDPT(perturber, gen_client, ext_client, epsilon=epsilon)
