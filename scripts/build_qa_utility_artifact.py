"""Build QA-builder v2 assertions from frozen ranker inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.train.qa_builder import (
    RELATION_TEACHER_MODEL,
    RELATION_TEACHER_PROVIDER,
    AciTaskAdapter,
    OpenRouterRelationTeacher,
    artifact_views,
    build_utility_artifact,
    freeze_v2_environment_from_legacy_arms,
    llm_prefilter_context_candidates,
    read_context_batch,
    read_context_set_batch,
    render_frozen_action_vector,
)
from cloak.train.qa_audit import build_environment_audit, write_audit_sidecars
from cloak.train.relation_support_gate import (
    RelationSupportCascade,
    build_informative_context_judge,
    build_medgemma_judge,
)
from cloak.train.reward import QA_BASE_URL

RELATION_SUPPORT_JUDGE_MODEL = "medgemma-4b-it"


def _medgemma_client():
    from cloak.llm import LLMClient

    return LLMClient(
        RELATION_SUPPORT_JUDGE_MODEL, base_url=QA_BASE_URL, api_key="none", temperature=0.0,
        # max_tokens is mandatory: the judges return a one-line JSON verdict, and without a cap
        # llama-server runs n_predict=-1 — a degenerate generation then rambles to the context
        # wall (~28k tokens, observed 2026-07-19), stalling the build for ~30min per call AND
        # failing the verdict parse, which accept_on_error turns into a silent fail-open accept.
        max_tokens=128,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _build_relation_support_escalator():
    """Recall-first, accept-only escalator for opportunity-miner cue-misses (MedGemma judge on the
    shared llama-swap endpoint)."""
    # ponytail: MedGemma-only (validated core); the optional accept-only MedNLI cost tier can be
    # added via mednli_entail= once per-candidate LLM latency is the bottleneck.
    return RelationSupportCascade(build_medgemma_judge(_medgemma_client()))


def _build_context_prefilter():
    """LLM set-call proposer of context-literal relation candidates (MedGemma, higher token budget
    than the judge -- enumerations are longer than a yes/no verdict). Augments the gazetteer; the
    escalator still decides acceptance."""
    from cloak.llm import LLMClient
    client = LLMClient(
        RELATION_SUPPORT_JUDGE_MODEL, base_url=QA_BASE_URL, api_key="x", temperature=0.0,
        max_tokens=256, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return lambda source, environment_document: llm_prefilter_context_candidates(
        source, environment_document, client.generate)


_READER_PIN_FIELDS = frozenset({
    "model", "endpoint", "prompt_version", "response_schema", "revision",
})


def _hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_manifest_reader_pin(manifest: Mapping) -> dict:
    reader_pin = manifest.get("reader_pin")
    if not isinstance(reader_pin, Mapping) or not reader_pin:
        raise ValueError("threshold manifest reader_pin must be a structured non-empty mapping")
    missing = sorted(_READER_PIN_FIELDS - set(reader_pin))
    if missing:
        raise ValueError(f"threshold manifest reader_pin is missing fields: {missing}")
    if any(
        reader_pin.get(field) is None
        or reader_pin.get(field) == ""
        or reader_pin.get(field) == {}
        for field in _READER_PIN_FIELDS
    ):
        raise ValueError("threshold manifest reader_pin fields must be non-empty")
    return dict(reader_pin)


def _selected_environment(environment: dict, corpus: str, doc_ids: list[str]) -> dict:
    available = environment.get("corpora", {}).get(corpus)
    if available is None:
        raise SystemExit(f"corpus {corpus!r} is absent from the ranker environment")
    missing = [doc_id for doc_id in doc_ids if doc_id not in available]
    if missing:
        raise SystemExit(f"documents absent from ranker environment: {missing}")
    return {
        **environment,
        "corpora": {corpus: {doc_id: available[doc_id] for doc_id in doc_ids}},
    }


def _source_rows(corpus: str, doc_ids: list[str]) -> dict[str, dict]:
    wanted = set(doc_ids)
    rows = {row["id"]: row for row in load_task_docs(corpus) if row["id"] in wanted}
    missing = [doc_id for doc_id in doc_ids if doc_id not in rows]
    if missing:
        raise SystemExit(f"documents absent from corpus {corpus!r}: {missing}")
    return rows


def _action_renderer(
    frozen_environment: dict,
    source_documents: dict[str, str],
):
    def render(doc_id: str, action_vector: dict[str, str]) -> str:
        frozen_document = frozen_environment["documents"][doc_id]
        return render_frozen_action_vector(
            source_documents[doc_id], frozen_document, action_vector,
        )[0]

    return render


def _relation_teacher_from_args(args, *, include_reasoning: bool = False) -> OpenRouterRelationTeacher:
    """Construct the GPT-OSS teacher from the CLI's routing args (prod entry point)."""
    return OpenRouterRelationTeacher(
        model=getattr(args, "teacher_model", RELATION_TEACHER_MODEL),
        routed_provider=(getattr(args, "teacher_provider", RELATION_TEACHER_PROVIDER) or None),
        allow_fallbacks=bool(getattr(args, "teacher_allow_fallbacks", False)),
        include_reasoning=include_reasoning,
    )


def build_from_files(
    args, *, relation_teacher=None, secondary_relation_teacher=None,
    reader=read_context_batch,
) -> dict:
    manifest = json.loads(Path(args.threshold_manifest).read_text())
    family_budgets = manifest.get("family_budgets")
    if not isinstance(family_budgets, dict) or set(family_budgets) != {"context", "delivered"}:
        raise SystemExit("threshold manifest requires context and delivered family_budgets")
    _require_manifest_reader_pin(manifest)

    environment = json.loads(Path(args.env).read_text())
    arms = json.loads(Path(args.arms).read_text())
    arms_meta = dict(arms.pop("_meta", {}) or {})
    environment_audit = dict(arms_meta.get("environment_audit", {}) or {})
    detector_pin = dict(arms_meta.get("detector", {}) or {})
    rows = _source_rows(args.corpus, args.doc_id)
    if any(not doc_id.startswith("aci/") for doc_id in rows):
        raise SystemExit("the implemented task adapter currently supports ACI documents only")
    source_documents = {doc_id: row["text"] for doc_id, row in rows.items()}
    persisted_frozen = environment.get("frozen_environment")
    if isinstance(persisted_frozen, Mapping):
        documents = persisted_frozen.get("documents")
        if not isinstance(documents, Mapping):
            raise SystemExit("ranker-v2 environment has no frozen documents")
        missing = [doc_id for doc_id in args.doc_id if doc_id not in documents]
        if missing:
            raise SystemExit(f"documents absent from frozen ranker-v2 environment: {missing}")
        frozen_environment = {
            "artifact_version": persisted_frozen.get("artifact_version"),
            "environment_hash": persisted_frozen.get("environment_hash"),
            "documents": {doc_id: documents[doc_id] for doc_id in args.doc_id},
        }
    else:
        environment = _selected_environment(environment, args.corpus, args.doc_id)
        frozen_environment = freeze_v2_environment_from_legacy_arms(
            environment, arms,
            detector_provenance=detector_pin or None,
            source_documents=source_documents,
        )
    environment_audit = build_environment_audit(
        frozen_environment, source_documents=source_documents,
    )
    references = {doc_id: row["gold_ref"] for doc_id, row in rows.items()}
    gleaning_requested = bool(
        getattr(args, "relation_teacher_gleaning", False)
        or os.getenv("CLOAK_RELATION_TEACHER_GLEANING") == "1"
    )
    if relation_teacher is None and (args.relation_teacher or gleaning_requested):
        relation_teacher = _relation_teacher_from_args(args)
    escalation_configured = bool(
        gleaning_requested and manifest.get("relation_escalation_policy") is not None
    )
    # The gleaning+repair pass reuses the primary GPT-OSS teacher config; the second
    # call differs only by prompt (relation_repair_prompt), so it is a distinct cache key.
    if secondary_relation_teacher is None and escalation_configured:
        # CLOAK_TEACHER_REASONING=include has the gleaning teacher return its reasoning trace for
        # prompt A/B tweaking. Scoped to the secondary teacher (its prompt already differs, so it is
        # a fresh cache key regardless); the primary teacher is untouched, so its cache still hits.
        secondary_relation_teacher = _relation_teacher_from_args(
            args, include_reasoning=os.getenv("CLOAK_TEACHER_REASONING", "").lower() == "include")
    teacher_pin = None
    if relation_teacher is not None:
        raw_teacher_pin = getattr(relation_teacher, "pin", None)
        if not isinstance(raw_teacher_pin, Mapping) or not raw_teacher_pin:
            raise ValueError("relation teacher requires an explicit non-empty pin")
        teacher_pin = dict(raw_teacher_pin)
    secondary_teacher_pin = None
    if secondary_relation_teacher is not None:
        raw_secondary_pin = getattr(secondary_relation_teacher, "pin", None)
        if not isinstance(raw_secondary_pin, Mapping) or not raw_secondary_pin:
            raise ValueError("secondary relation teacher requires an explicit non-empty pin")
        secondary_teacher_pin = dict(raw_secondary_pin)

    pins = {
        key: value for key, value in manifest.items()
        if key not in {
            "family_budgets",
            "min_context_assertions",
            "min_contextual_relation_assertions",
            "reader_threshold",
        }
    }
    pins.update({
        "gate_manifest_hash": _hash(manifest),
        "task_pin": AciTaskAdapter.task_pin,
        # v14: deterministic reverse-orientation ambiguity recovery (Sources 1+2) -- an ambiguous
        # (relation, subject) group flips every object (teacher-proposed + judge-accepted) to an
        # answer_role=subject QA, gated in an isolated doc-global pass (see docs/handoffs/
        # 2026-07-20-qa-relation-ambiguity-and-open-tasks.md). v13: semantic_property probes disabled.
        "builder_pin": "qa-builder-v2-assertion-compiler-v16",
        "detector_pin": detector_pin or None,
        "teacher_pin": teacher_pin,
        "environment_audit_hash": environment_audit.get("audit_hash") or None,
        "source_hashes": {doc_id: _hash(text) for doc_id, text in source_documents.items()},
        "reference_hashes": {doc_id: _hash(text) for doc_id, text in references.items()},
    })
    if escalation_configured and secondary_teacher_pin is not None:
        pins.update({
            "relation_teacher_pins": {
                "primary": teacher_pin,
                "secondary": secondary_teacher_pin,
            },
            "relation_escalation_policy": dict(manifest["relation_escalation_policy"]),
        })
    relation_support_escalator = (
        _build_relation_support_escalator()
        if getattr(args, "relation_support_escalation", False) else None
    )
    if relation_support_escalator is not None:
        pins["relation_support_escalation"] = {
            "judge_model": RELATION_SUPPORT_JUDGE_MODEL, "base_url": QA_BASE_URL,
        }
    informative_context_judge = (
        build_informative_context_judge(_medgemma_client())
        if getattr(args, "informative_context_judge", False) else None
    )
    if informative_context_judge is not None:
        pins["informative_context_judge"] = {
            "judge_model": RELATION_SUPPORT_JUDGE_MODEL, "base_url": QA_BASE_URL,
        }
    context_prefilter = None
    if getattr(args, "relation_support_prefilter", False):
        if relation_support_escalator is None:
            raise SystemExit(
                "--relation-support-prefilter requires --relation-support-escalation: the prefilter's "
                "literal candidates are cue-misses that only the judge can accept.")
        context_prefilter = _build_context_prefilter()
        pins["relation_support_prefilter"] = {
            "model": RELATION_SUPPORT_JUDGE_MODEL, "base_url": QA_BASE_URL,
        }
    deterministic_relation_stage = bool(getattr(args, "relation_deterministic_stage", False))
    if deterministic_relation_stage:
        pins["relation_deterministic_stage"] = True
    finer_level_check = getattr(args, "reader_finer_level_check", None) or None
    if finer_level_check:
        pins["reader_finer_level_check"] = finer_level_check
    return build_utility_artifact(
        frozen_environment,
        AciTaskAdapter(references),
        source_documents,
        threshold_manifest=manifest,
        pins=pins,
        reader=reader,
        render_action_vector=_action_renderer(
            frozen_environment, source_documents
        ),
        relation_teacher=relation_teacher,
        secondary_relation_teacher=(
            secondary_relation_teacher if escalation_configured else None
        ),
        environment_audit=environment_audit or None,
        relation_support_escalator=relation_support_escalator,
        informative_context_judge=informative_context_judge,
        context_prefilter=context_prefilter,
        deterministic_relation_stage=deterministic_relation_stage,
        set_reader=read_context_set_batch if reader is read_context_batch else None,
        finer_level_check=finer_level_check,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--corpus", default="clinical")
    parser.add_argument("--doc-id", action="append", required=True)
    parser.add_argument("--threshold-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--relation-teacher",
        action="store_true",
        help="enable the primary cached GPT-OSS relation teacher",
    )
    parser.add_argument(
        "--teacher-model",
        default=RELATION_TEACHER_MODEL,
        help=f"OpenRouter model id for the relation teacher (default {RELATION_TEACHER_MODEL})",
    )
    parser.add_argument(
        "--teacher-provider",
        default=RELATION_TEACHER_PROVIDER,
        help=(
            "OpenRouter routed provider for the teacher, e.g. deepinfra/turbo, deepinfra/bf16 "
            f"(default {RELATION_TEACHER_PROVIDER}); pass an empty string to leave routing "
            "to OpenRouter. The routed provider is part of the teacher pin and the cache key."
        ),
    )
    parser.add_argument(
        "--teacher-allow-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow OpenRouter to fall back off the routed provider (default off)",
    )
    parser.add_argument(
        "--relation-teacher-gleaning",
        action="store_true",
        help=(
            "with a manifest relation_escalation_policy, conditionally run one "
            "GPT-OSS gleaning+repair pass after the primary GPT-OSS build "
            "(targets ambiguous / fixable-rejected / missed relations)"
        ),
    )
    parser.add_argument(
        "--relation-support-escalation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "recover opportunity-miner cue-misses with the MedGemma judge (recall-first). "
            "ON by default; use --no-relation-support-escalation to fall back to the cue gate. "
            "Requires the medgemma-4b-it model on the llama-swap endpoint."
        ),
    )
    parser.add_argument(
        "--informative-context-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "admit semantic_property context probes on a role-cue regex miss: free when the "
            "entity already carries mined relation evidence, else one MedGemma informativeness "
            "call on its redacted locator sentence. ON by default; "
            "--no-informative-context-judge falls back to the pure regex lexicon."
        ),
    )
    parser.add_argument(
        "--relation-support-prefilter",
        action="store_true",
        help=(
            "augment the gazetteer context-literal candidates with an LLM set-call prefilter that "
            "recovers drug/symptom/condition literal relation objects the gazetteer cannot emit. "
            "Opt-in (off by default); requires --relation-support-escalation. Augment-only, so it "
            "cannot drop a currently-accepted pair. Adds ~(#conditions x 5) MedGemma calls per doc."
        ),
    )
    parser.add_argument(
        "--relation-deterministic-stage",
        action="store_true",
        help=(
            "generate template relation QAs from mined opportunity pairs between the primary "
            "teacher pass and gleaning (no teacher call): forward with reverse fallback for "
            "singleton span pairs, reverse-only for ambiguous groups, literal-reverse from the "
            "all-accepted seed; answer levels searched coarsest-to-finest against the reader gate. "
            "Facts kept here are excluded from the paid gleaning targets. Opt-in (off by default)."
        ),
    )
    parser.add_argument(
        "--reader-finer-level-check",
        nargs="?",
        const="hard",
        choices=("hard", "soft"),
        default=None,
        help=(
            "reward-band certification: for every gate-passing relation QA, re-render the "
            "representative with the answer decision at each level FINER than its supported "
            "level and re-read. 'hard' (bare flag): an unreadable finer level REJECTS the QA, "
            "routed lattice_suspect so it never reaches repair/glean. 'soft': keep the QA, only "
            "record the scores and emit the finer-level-failures worklist. Adds ~one local "
            "reader call per gate-passing row per finer level. Opt-in."
        ),
    )
    return parser.parse_args(argv)


def write_finer_level_failures(
    artifact: dict, source_documents: Mapping[str, str], output: Path,
) -> tuple[Path, int]:
    """Lattice-producer worklist from the reward-band certification: one JSONL row per kept
    relation QA with >=1 UNREADABLE finer answer level. Each row names the answer decision's
    profile, the finer levels that failed/passed the reader, the relation (type, subject,
    object, supported level), the question, and the doc_orig excerpt the reader actually saw."""
    from cloak.train.qa_builder import _reader_excerpt

    threshold = float(artifact.get("reader_threshold") or 1.0)

    def _rows_with_finer_levels():
        # kept assertions (soft mode records on validation evidence) ...
        for row in artifact.get("assertions", {}).values():
            if row.get("subtype") != "contextual_relation":
                continue
            evidence = row.get("evidence") or {}
            finer = (evidence.get("validation") or {}).get("finer_levels") or {}
            if finer:
                yield row, row.get("question"), row.get("answer_target") or {}, evidence, finer
        # ... and hard-mode rejections (the QA never became an assertion; the reject site
        # stashed question/answer_target/finer_levels on the rejection evidence)
        for row in (artifact.get("rejections") or {}).get("records", []):
            evidence = row.get("evidence") or {}
            finer = evidence.get("finer_levels") or {}
            if finer:
                yield (row, evidence.get("question"), evidence.get("answer_target") or {},
                       evidence, finer)

    def _level_entry(value):
        # per-level value is {"score", "render"} (current) or a bare float (older artifacts)
        if isinstance(value, dict):
            return float(value.get("score", 0.0)), str(value.get("render") or "ok")
        return float(value), "ok"

    def _provenance(row, evidence, arguments, answer_role):
        run_id = evidence.get("run_id")
        group_id = str(row.get("group_id") or "")
        context_count = sum(1 for a in arguments if a.get("kind") == "context")
        if run_id in (None, "primary"):
            return "teacher_primary"
        if run_id in ("gleaning", "reverse_framing", "literal_reverse"):
            return {"gleaning": "teacher_gleaning",
                    "reverse_framing": "reverse_framing_flip",
                    "literal_reverse": "literal_reverse"}[run_id]
        if run_id == "deterministic_stage":
            if group_id.startswith("set_forward:"):
                return "stage_set_forward"
            if group_id.startswith("compound_span_reverse:"):
                return "stage_compound_span_reverse"
            if group_id.startswith("literal_reverse:") or context_count:
                return ("stage_literal_reverse_compound" if context_count > 1
                        else "stage_literal_reverse_single")
            return ("stage_span_reverse" if answer_role == "subject"
                    else "stage_span_forward")
        return str(run_id)

    rows_out: list[str] = []
    for row, question, target, evidence, finer_levels in _rows_with_finer_levels():
        entries = {level: _level_entry(value) for level, value in finer_levels.items()}
        rejected = sorted(level for level, (score, _r) in entries.items() if score < threshold)
        if not rejected:
            continue
        # levels_accepted = the VERIFIED-accepted part of the ladder: the supported level (it
        # passed the full three-point gate) plus any finer level the band check confirmed.
        # Coarser-than-supported levels are deliberately absent: the stage's level search tried
        # and gate-FAILED them (that is why the supported level is the coarsest), and teacher
        # rows never tested them -- listing them as accepted would be fabrication.
        supported_level = str(target.get("required_property") or "")
        accepted = sorted(
            {level for level, (score, _r) in entries.items() if score >= threshold}
            | ({supported_level} if supported_level else set())
        )
        doc_id = str(row.get("doc_id"))
        document = artifact["documents"][doc_id]
        decision = next(
            (d for d in document.get("decisions", [])
             if str(d.get("decision_id")) == str(target.get("decision_id"))), {})
        occurrences = {str(o["occurrence_id"]): o for o in document.get("occurrences", [])}

        def describe(argument):
            if argument.get("kind") == "context":
                return str(argument.get("literal") or "")
            occurrence = occurrences.get(str(argument.get("occurrence_id"))) or {}
            return str(occurrence.get("surface") or argument.get("surface") or "")

        def _argument_decision(argument):
            occurrence = occurrences.get(str(argument.get("occurrence_id"))) or {}
            return str(occurrence.get("decision_id") or "")

        arguments = list(evidence.get("arguments") or [])
        subject = next((a for a in arguments if a.get("role") == "subject"), {})
        objects = [a for a in arguments if a.get("role") == "object"]
        # orientation from the ANSWER decision (answer_role is not persisted on compiled rows)
        answer_role = ("subject" if _argument_decision(subject) == str(target.get("decision_id"))
                       else "object")
        rows_out.append(json.dumps({
            "profile": {
                "key": str(decision.get("canonical_key") or ""),
                "runtime_type": str(decision.get("runtime_type") or ""),
            },
            "levels_rejected": rejected,
            "levels_accepted": accepted,
            # what stopped each rejected level: "read_failed" (probe rendered; the reader's
            # answer did not resolve/entail) vs "no_joint_arm" (level-fill collision: the probe
            # could not even be rendered jointly -- a render limitation, not a bad level)
            "rejection_causes": {
                level: ("no_joint_arm" if entries[level][1] == "no_joint_arm" else "read_failed")
                for level in rejected
            },
            "relation": {
                "type": row.get("relation"),
                "provenance": _provenance(row, evidence, arguments, answer_role),
                "answer_role": answer_role,
                "subject": describe(subject),
                "object": (describe(objects[0]) if len(objects) == 1
                           else [describe(a) for a in objects]),
                "supported_level": supported_level,
            },
            "question": question,
            "doc_id": doc_id,
            "doc_context": _reader_excerpt(source_documents.get(doc_id, ""), evidence),
        }, sort_keys=True))
    path = output.with_name(f"{output.stem}.finer-level-failures.jsonl")
    path.write_text("\n".join(rows_out) + ("\n" if rows_out else ""))
    return path, len(rows_out)


def write_artifacts(artifact: dict, output: Path) -> tuple[Path, Path]:
    assertions_view, qa_pairs_view = artifact_views(artifact)
    assertions_output = output.with_name(f"{output.stem}.assertions.json")
    qa_pairs_output = output.with_name(f"{output.stem}.qa-pairs.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=1))
    assertions_output.write_text(json.dumps(assertions_view, indent=1))
    qa_pairs_output.write_text(json.dumps(qa_pairs_view, indent=1))
    qa_audit = artifact.get("qa_audit")
    if isinstance(qa_audit, Mapping):
        write_audit_sidecars(
            qa_audit, output.with_name(f"{output.stem}.qa-audit"),
        )
    return assertions_output, qa_pairs_output


def main(
    argv=None, *, relation_teacher=None, secondary_relation_teacher=None,
    reader=read_context_batch,
):
    args = parse_args(argv)
    artifact = build_from_files(
        args,
        relation_teacher=relation_teacher,
        secondary_relation_teacher=secondary_relation_teacher,
        reader=reader,
    )
    output = Path(args.out)
    assertions_output, qa_pairs_output = write_artifacts(artifact, output)
    qa_audit_base = output.with_name(f"{output.stem}.qa-audit")
    qa_audit_paths = [
        qa_audit_base.with_name(qa_audit_base.name + ".json"),
        qa_audit_base.with_name(qa_audit_base.name + ".jsonl"),
        qa_audit_base.with_name(qa_audit_base.name + ".md"),
    ]
    print(
        f"wrote {output}: docs={len(artifact['documents'])} "
        f"assertions={len(artifact['assertions'])} "
        f"rejections={sum(artifact['rejections']['summary_by_reason'].values())}; "
        f"views={assertions_output},{qa_pairs_output}; "
        f"qa_audit={','.join(map(str, qa_audit_paths))}",
        flush=True,
    )
    if getattr(args, "reader_finer_level_check", False):
        source_documents = {
            doc_id: row["text"] for doc_id, row in _source_rows(args.corpus, args.doc_id).items()
        }
        failures_path, failure_count = write_finer_level_failures(
            artifact, source_documents, output,
        )
        print(f"finer-level failures (lattice worklist): {failure_count} -> {failures_path}",
              flush=True)


if __name__ == "__main__":
    main()
