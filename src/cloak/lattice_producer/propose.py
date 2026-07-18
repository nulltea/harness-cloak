"""Local-model proposal helpers and bounded context packet assembly."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from cloak.lattice_producer.io import atomic_write_json
from cloak.lattice_producer.vocabulary import CanonicalVocabulary

QWEN36_MODEL = "Qwen3.6-35B-A3B"
QWEN36_THINKING_BUDGET_TOKENS = 2048

_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)


def _has_choices(response) -> bool:
    return bool(getattr(response, "choices", None))


def _response_content(response) -> str:
    """Content of the first choice, or "{}" when the provider returned no usable choices (a free
    endpoint can return an error/empty payload with choices=None that doesn't raise)."""
    if not _has_choices(response):
        return "{}"
    return response.choices[0].message.content or "{}"


def _create_with_retry(client, *, model, request_kwargs, attempts=4, base_timeout=600):
    """Bounded retry around a single chat completion. Escalates the per-call timeout each attempt
    and sleeps a short backoff between tries, so a transient 429/timeout OR an empty (no-choices)
    payload from a rate-limited free endpoint rides through instead of killing the run. On a real
    transport error it re-raises after `attempts`; on a persistently empty response it returns that
    response so the caller degrades to a no-candidates diagnostic rather than crashing."""
    last_exc = None
    last_empty = None
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model, timeout=base_timeout * (attempt + 1), **request_kwargs
            )
        except _RETRYABLE as exc:
            last_exc = exc
        else:
            if _has_choices(response):
                return response
            last_empty = response
        if attempt < attempts - 1:
            time.sleep(min(5.0 * (attempt + 1), 30.0))
    if last_empty is not None:
        return last_empty
    raise last_exc


def _load_profiles(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"profiles": {}}
    return json.loads(path.read_text())


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# Per-runtime-type level guidance. Each entry names the SPECIFIC nearest tier for that type and
# a concrete worked chain, so the model sees only the rule relevant to the item it is processing
# rather than one prompt with every type's examples jammed together. `_DEFAULT_LEVEL_GUIDANCE`
# covers any runtime type without a bespoke entry (profession, LOC, ORG, ...).
_DEFAULT_LEVEL_GUIDANCE = (
    "The FIRST (nearest) level must be the MOST SPECIFIC truthful category the surface directly "
    "belongs to -- its immediate parent class, never a broad catch-all. Each subsequent level "
    "widens by exactly ONE step; do not skip real intermediate tiers or jump to a universal "
    "catch-all. Prefer 3-4 tiers when genuine intermediate categories exist."
)
_TYPE_LEVEL_GUIDANCE: dict[str, str] = {
    "drug": (
        "This entry is a DRUG. The nearest level is its specific pharmacologic/mechanistic class, "
        "e.g. metoprolol -> 'beta blocker' (or 'selective beta-1 adrenergic antagonist'), then "
        "widen one step at a time: -> 'antihypertensive agent' -> 'cardiovascular agent'. Do NOT "
        "start at 'medication', 'drug', or 'pharmaceutical compound' -- those are only the broadest "
        "tier, never the first. proposed_count is the number of DISTINCT drugs in that class "
        "(a specific class holds ~dozens; the broadest tier at most low thousands)."
    ),
    "health-condition": (
        "This entry is a HEALTH CONDITION. The nearest level is its specific disease family, e.g. "
        "dermatitis -> 'eczematous skin disorder' -> 'inflammatory skin disease' -> 'skin disease'. "
        "Do NOT start at 'medical condition', 'disease', or 'disorder' -- those are only the "
        "broadest tier. proposed_count is the number of DISTINCT conditions in that family "
        "(a specific family holds ~dozens; the broadest tier at most low thousands)."
    ),
    "medical-procedure": (
        "This entry is a MEDICAL PROCEDURE. The nearest level is its specific procedure class, e.g. "
        "upper endoscopy -> 'esophagogastroduodenoscopy' -> 'upper gastrointestinal endoscopy' -> "
        "'endoscopic procedure'. Do NOT start at 'medical procedure', 'clinical service', or "
        "'human activity' -- those are only the broadest tier. proposed_count is the number of "
        "DISTINCT procedures in that class (a specific class holds ~dozens)."
    ),
}


def _level_guidance_for(runtime_type: str) -> str:
    return _TYPE_LEVEL_GUIDANCE.get(runtime_type, _DEFAULT_LEVEL_GUIDANCE)


# Reason-specific repair instructions. The generic text stays as the fallback for any future
# reprocess_reason without a bespoke entry.
_DEFAULT_REPROCESS_INSTRUCTION = (
    "This entry is being reprocessed from audit evidence: verify the diagnostic against "
    "the marked context and source-backed candidates. Do not blindly copy the existing "
    "profile or assume the audit is correct; address the stated failure while preserving "
    "truthful levels and anonymity-set count semantics."
)
_REPROCESS_INSTRUCTIONS: dict[str, str] = {
    "missing_profile_reference_backed": (
        "Repair task: this surface appeared in a real document, a trusted local source resolves "
        "it, but no profile exists. Verify from marked_context_sentence that it names ONE stable "
        "real-world referent of this runtime_type, then build the ladder from that referent's "
        "truthful hierarchy. If the context leaves several referents equally plausible, set "
        "surface_confidence to \"low\" or \"ambiguous\" instead of picking one."
    ),
    "missing_profile_type_entailed": (
        "Repair task: this surface appeared in a real document with no existing profile. First "
        "verify from marked_context_sentence that it names ONE stable real-world referent of "
        "this runtime_type. If it is a generic class mention, detector residue, or has several "
        "equally plausible referents, set surface_confidence to \"low\" or \"ambiguous\" "
        "instead of inventing a ladder."
    ),
    "lattice_structure_or_utility_failure": (
        "Repair task: this existing profile failed in use. current_profile is the ladder under "
        "repair; repair_findings lists the observed failures (e.g. a level a reader could not "
        "use while a coarser level in the same chain worked, non-monotone or jumpy counts, a "
        "degenerate one-rung menu). Rebuild the ladder truthfully: re-phrase, replace, or "
        "re-position the failing rungs and add genuine intermediate tiers. Propose fresh counts "
        "per count_semantics_instruction -- do not copy current_profile's levels or counts "
        "unchanged, and do not fix counts by flattening the ladder."
    ),
}


def assemble_context_packet(
    item: dict[str, Any],
    *,
    profiles_path: str | Path,
    run_dir: str | Path,
    prompt_version: str,
    max_context_rows: int,
    proposed_out: str | Path | None = None,
) -> dict[str, Any]:
    runtime_type = item.get("runtime_type")
    detector_label = item.get("detector_label_family")
    surface = str(item.get("surface") or item.get("canonical_value") or "")
    artifact = _load_profiles(profiles_path)
    entries = artifact.get("profiles", {}).get(runtime_type, {})
    relevant = []
    surface_tokens = {tok for tok in re.split(r"\W+", surface.lower()) if tok}
    for canonical, row in sorted(entries.items()):
        aliases = [str(a) for a in row.get("aliases", [])]
        haystack = " ".join([canonical, *aliases]).lower()
        if canonical == surface.lower() or surface.lower() in haystack or surface_tokens.intersection(re.split(r"\W+", haystack)):
            relevant.append(
                {
                    "canonical_value": canonical,
                    "aliases": aliases[:3],
                    "levels": list(row.get("levels", []))[:5],
                    "source_ids": list(row.get("source_ids", []))[:3],
                }
            )
    relevant = relevant[:max_context_rows]
    vocabulary = CanonicalVocabulary(str(runtime_type), proposed_out=proposed_out) if runtime_type else None
    vocabulary_slice = vocabulary.context_slice(n=max_context_rows, surface=surface) if vocabulary else []
    packet = {
        "prompt_version": prompt_version,
        "task_kind": item.get("task_kind", "level-proposal"),
        "runtime_type": runtime_type,
        "detector_label_family": detector_label,
        "surface_or_entry": surface,
        "marked_context_sentence": item.get("marked_context_sentence", ""),
        "type_policy": "Produce truthful grammatical generalization levels only; never emit placeholders or direct identifiers.",
        "allowed_outputs": "Strict JSON with entry aliases and ordered candidate levels. Each level must include a proposed_count, count_evidence, selector, and rationale. Counts are review evidence, not certifying source-backed counts.",
        "required_proposal_fields": ["aliases", "candidates", "surface_confidence", "type_membership"],
        "type_membership_instruction": (
            "type_membership must be \"confirmed\" or \"mismatch\". Report \"mismatch\" when the "
            "surface does not denote an entity of runtime_type at all (e.g. a procedure name or "
            "report artifact queued as a health condition) -- never invent a ladder for a "
            "mis-typed surface."
        ),
        "surface_confidence_instruction": (
            "surface_confidence must be \"high\", \"low\", or \"ambiguous\". A short (<=4 character) "
            "or multi-referent clinical/domain abbreviation must be marked \"low\" or \"ambiguous\" "
            "unless marked_context_sentence clearly disambiguates it -- do not silently pick one "
            "referent among several equally plausible ones and report \"high\" anyway."
        ),
        "required_level_fields": ["level", "proposed_count", "count_evidence", "selector", "rationale", "reused_canonical_label"],
        "min_levels": 2,
        "count_semantics_instruction": (
            "proposed_count is an ANONYMITY-SET SIZE: the number of DISTINCT real-world entities "
            "of this runtime_type that also belong to this level. It is NOT people affected, NOT "
            "prevalence, NOT disease burden, NOT sales or market size -- a count in the millions "
            "or billions is always wrong here. It increases monotonically as levels get broader: "
            "the nearest specific level holds only dozens to a few hundred sibling entities, and "
            "only the broadest levels reach the low thousands."
        ),
        "level_guidance": _level_guidance_for(str(runtime_type or "")),
        "nearby_profile_rows": relevant,
        "canonical_vocabulary_slice": vocabulary_slice,
        "canonical_vocabulary_instruction": (
            "canonical_vocabulary_slice lists {label, count} rows this run already uses. If any "
            "label fits a proposed level, reuse it verbatim, set reused_canonical_label: true, "
            "and reuse its attached count. Only coin new phrasing when nothing fits."
        ),
        "category_slice": [],
        "forbidden_outputs": ["type-name phrases", "original surface leaks", "direct identifiers"],
    }
    if item.get("reprocess_reason"):
        reason = str(item["reprocess_reason"])
        packet["reprocess_reason"] = reason
        packet["reprocess_hint"] = str(item.get("reprocess_hint", ""))
        packet["reprocess_instruction"] = _REPROCESS_INSTRUCTIONS.get(reason, _DEFAULT_REPROCESS_INSTRUCTION)
        findings = item.get("repair_findings")
        if isinstance(findings, list) and findings:
            packet["repair_findings"] = findings[:5]
        # A ladder repair must see the ladder it is repairing -- nearby_profile_rows truncates
        # levels and omits counts, which are exactly what a structure/utility failure is about.
        current_row = entries.get(str(item.get("canonical_value") or "")) or entries.get(surface)
        if isinstance(current_row, dict) and current_row.get("levels"):
            groundings = current_row.get("level_grounding") or {}
            packet["current_profile"] = {
                "canonical_value": str(item.get("canonical_value") or surface),
                "aliases": [str(alias) for alias in current_row.get("aliases", [])][:6],
                "levels": list(current_row.get("levels", [])),
                "level_counts": dict(current_row.get("level_counts") or {}),
                "count": current_row.get("count"),
                "level_grounding_status": {
                    level: (grounding or {}).get("status")
                    for level, grounding in groundings.items()
                    if isinstance(grounding, dict)
                },
            }
    if item.get("retry_attempt"):
        packet["retry_attempt"] = int(item.get("retry_attempt", 0))
        packet["previous_rejection_feedback"] = list(item.get("rejection_feedback", []))
        packet["retry_instruction"] = (
            "The previous proposal failed gates. Each previous_rejection_feedback item names the "
            "rung that failed and, where available, a repair_hint with the specific correction -- "
            "apply every repair_hint. Keep truthful levels, non-flat counts, and concrete count "
            "evidence."
        )
    packet["artifact_slice_hashes"] = {"nearby_profile_rows": _hash_payload(relevant)}
    packet["context_packet_hash"] = _hash_payload(packet)
    return packet


# Hosts the producer is allowed to send prompts to. localhost is the default (llama-swap on the
# GPU); openrouter.ai is an explicit opt-in for offline lattice production against a hosted model
# (the prompts are PII *type* surfaces like "metoprolol", not user documents). Anything else is
# blocked so a stray base_url can't silently exfiltrate prompts.
_ALLOWED_REMOTE_HOSTS = {"openrouter.ai"}


def is_openrouter(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") == "openrouter.ai"


def ensure_local_base_url(base_url: str) -> None:
    host = urlparse(base_url).hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"} or host in _ALLOWED_REMOTE_HOSTS:
        return
    raise ValueError(f"base URL must be localhost or an allowed remote ({_ALLOWED_REMOTE_HOSTS}), got {base_url}")


def extract_candidate_levels(payload: dict[str, Any]) -> list[dict[str, Any]]:
    payload_aliases = [str(alias).strip().lower() for alias in payload.get("aliases", []) if str(alias).strip()]
    if isinstance(payload.get("candidates"), list):
        candidates = []
        for candidate in payload["candidates"]:
            if not isinstance(candidate, dict) or not str(candidate.get("level", "")).strip():
                continue
            out = {
                **candidate,
                "level": str(candidate.get("level", "")).strip(),
                "source_family": candidate.get("source_family", "model-proposed"),
            }
            aliases = candidate.get("aliases") if isinstance(candidate.get("aliases"), list) else payload_aliases
            if aliases:
                out["aliases"] = [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
            candidates.append(out)
        return candidates
    for key in ("candidate_levels", "lattice_generalization_levels"):
        values = payload.get(key)
        if isinstance(values, list):
            selectors = payload.get("grounding_selectors") if isinstance(payload.get("grounding_selectors"), list) else []
            return [
                {
                    "level": str(level).strip(),
                    "selector": selectors[idx] if idx < len(selectors) else key,
                    "source_family": "model-proposed",
                    **({"aliases": payload_aliases} if payload_aliases else {}),
                }
                for idx, level in enumerate(values)
                if str(level).strip()
            ]
    proposed_levels = payload.get("proposed_levels") or payload.get("lattice_proposals")
    if isinstance(proposed_levels, list):
        candidates = []
        for record in proposed_levels:
            if isinstance(record, dict) and str(record.get("level", "")).strip():
                selectors = record.get("grounding_selectors") or record.get("selectors") or []
                candidates.append(
                    {
                        **record,
                        "level": str(record["level"]).strip(),
                        "selector": selectors[0] if selectors else record.get("selector", "proposed_levels"),
                        "source_family": "model-proposed",
                        **({"aliases": payload_aliases} if payload_aliases and not record.get("aliases") else {}),
                    }
                )
            elif str(record).strip():
                candidate = {"level": str(record).strip(), "selector": "proposed_levels", "source_family": "model-proposed"}
                if payload_aliases:
                    candidate["aliases"] = payload_aliases
                candidates.append(candidate)
        return candidates
    return []


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / "lattice_producer" / f"{key}.json"


def _parse_model_json(content: str) -> dict[str, Any]:
    raw = content or "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"candidates": [], "parse_error": "invalid_json", "raw": raw}
    if isinstance(payload, dict):
        return payload
    return {"candidates": [], "parse_error": "non_object_json", "raw": raw}


def propose_with_llama_swap(
    item: dict[str, Any],
    *,
    profiles_path: str | Path,
    run_dir: str | Path,
    prompt_version: str,
    max_context_rows: int,
    base_url: str,
    model: str,
    escalation_model: str | None = None,
    thinking_budget_tokens: int = -1,
    proposed_out: str | Path | None = None,
) -> dict[str, Any]:
    ensure_local_base_url(base_url)
    packet = assemble_context_packet(
        item,
        profiles_path=profiles_path,
        run_dir=run_dir,
        prompt_version=prompt_version,
        max_context_rows=max_context_rows,
        proposed_out=proposed_out,
    )
    identity = {
        "model": model,
        "base_url": base_url,
        "prompt_version": prompt_version,
        "item_id": item.get("item_id"),
        "context_packet_hash": packet["context_packet_hash"],
    }
    if is_openrouter(base_url) and os.environ.get("OPENROUTER_PROVIDER"):
        # a provider/quantization pin changes which weights answer -- never share cache entries
        # across pins (unpinned runs keep their historical cache keys)
        identity["openrouter_provider"] = os.environ["OPENROUTER_PROVIDER"]
    cache_key = _hash_payload(identity)
    cache_dir = Path(os.environ.get("CLOAK_LLM_CACHE", "data/llm_cache"))
    cache_file = _cache_path(cache_dir, cache_key)
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    if is_openrouter(base_url):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be set to use the openrouter.ai base URL")
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "local")
    client = OpenAI(base_url=base_url, api_key=api_key)
    repair_preamble = (
        "REPAIR MODE: this item reprocesses audit evidence. Read reprocess_instruction, "
        "repair_findings, and current_profile in the packet before proposing.\n"
        if packet.get("reprocess_reason")
        else ""
    )
    prompt = (
        repair_preamble
        + "Return strict JSON only. Propose a reviewable lattice profile row for this item. "
        "Include aliases for the entry, then AT LEAST TWO ordered candidate levels from nearest "
        "to broadest. Follow the packet's level_guidance (specific to this runtime_type) for how "
        "specific the nearest level must be, and count_semantics_instruction for proposed_count. "
        "Give count_evidence, selector, and rationale per level; counts are review evidence, not "
        "certified.\n\n"
        + json.dumps(packet, sort_keys=True)
    )
    request_kwargs = {
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if is_openrouter(base_url):
        # OpenRouter's reasoning knob (not the local llama-swap thinking_budget_tokens field).
        request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        # optional endpoint pin, e.g. OPENROUTER_PROVIDER="deepinfra/bf16" -> DeepInfra bf16
        # only, no fallback to other providers/quantizations
        provider = os.environ.get("OPENROUTER_PROVIDER", "")
        if provider:
            slug, _, quantization = provider.partition("/")
            routing: dict[str, Any] = {"order": [slug], "allow_fallbacks": False}
            if quantization:
                routing["quantizations"] = [quantization]
            request_kwargs["extra_body"]["provider"] = routing
    elif thinking_budget_tokens >= 0:
        request_kwargs["extra_body"] = {"thinking_budget_tokens": thinking_budget_tokens}
    response = _create_with_retry(client, model=model, request_kwargs=request_kwargs)
    payload = _parse_model_json(_response_content(response))
    if payload.get("parse_error") and escalation_model:
        response = _create_with_retry(client, model=escalation_model, request_kwargs=request_kwargs)
        payload = _parse_model_json(_response_content(response))
    payload["cache_key"] = cache_key
    payload["context_packet_hash"] = packet["context_packet_hash"]
    payload["model_used"] = model
    atomic_write_json(cache_file, payload)
    return payload
