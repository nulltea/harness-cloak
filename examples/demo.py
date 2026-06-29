"""End-to-end InferDPT demo on one document. Run: PYTHONPATH=src python examples/demo.py"""

from inferdpt import default

DOC = (
    "Sergio Garcia, a 36-year-old undocumented immigrant in California, has held two "
    "lifelong dreams: to become a U.S. citizen and to practice law. He has been waiting "
    "nineteen years for a visa still stuck in a backlog."
)

if __name__ == "__main__":
    pipe = default(epsilon=3.0)
    r = pipe.run(DOC, seed=0)
    for label, text in [
        ("DOC (private)", r.doc),
        ("DOC_p (sent to remote)", r.doc_p),
        ("GEN_p (remote output)", r.gen_p),
        ("OUTPUT (final)", r.output),
    ]:
        print(f"\n=== {label} ===\n{text}")
