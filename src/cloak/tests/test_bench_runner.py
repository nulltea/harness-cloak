from bench.runner import build_prompt, run_item, run_suite
from bench.schema import BenchmarkConfig
from bench.registry import load_items


class StubRemote:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _config(policy="all_placeholder", limit=1):
    return BenchmarkConfig(
        suite="primary_utility",
        limit=limit,
        seed=0,
        detector_version="gold",
        substitutor_version=policy,
        privacy_setting="tau=0.02",
        remote_model="stub-remote",
        extractor_version="current",
        attacker_version="offline-v1",
        output_dir="results/roundtrip_benchmark/test",
    )


def test_build_prompt_uses_private_doc_not_original_doc():
    item = load_items("primary_utility", limit=1, seed=0)[0]

    prompt = build_prompt(item, "PRIVATE DOC")

    assert "PRIVATE DOC" in prompt
    assert item.doc_orig not in prompt


def test_run_item_writes_stage_outputs_with_stub_remote():
    item = load_items("primary_utility", limit=1, seed=0)[0]
    remote = StubRemote(["<PERSON_1> is a patient."])

    trace = run_item(item, _config(), remote=remote)

    assert trace.stage.doc_p
    assert trace.stage.out_p == "<PERSON_1> is a patient."
    assert trace.stage.out_final
    assert remote.prompts and item.doc_orig not in remote.prompts[0]
    assert trace.config_hash == _config().config_hash()


def test_run_suite_uses_one_remote_call_per_item():
    remote = StubRemote(["summary one", "summary two"])

    traces = run_suite(_config(limit=2), remote=remote)

    assert len(traces) == 2
    assert len(remote.prompts) == 2
    assert all(trace.stage.out_final for trace in traces)
