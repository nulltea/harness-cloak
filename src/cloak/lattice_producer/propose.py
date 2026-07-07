"""Local-model proposal helpers and bounded context packet assembly."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from cloak.lattice_producer.io import atomic_write_json


def _load_profiles(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"profiles": {}}
    return json.loads(path.read_text())


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def assemble_context_packet(
    item: dict[str, Any],
    *,
    profiles_path: str | Path,
    run_dir: str | Path,
    prompt_version: str,
    max_context_rows: int,
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
    packet = {
        "prompt_version": prompt_version,
        "task_kind": item.get("task_kind", "level-proposal"),
        "runtime_type": runtime_type,
        "detector_label_family": detector_label,
        "surface_or_entry": surface,
        "marked_context_sentence": item.get("marked_context_sentence", ""),
        "type_policy": "Produce truthful grammatical generalization levels only; never emit placeholders or direct identifiers.",
        "allowed_outputs": "Strict JSON with candidate levels and grounding selectors; model-provided numeric counts are ignored.",
        "nearby_profile_rows": relevant,
        "category_slice": [],
        "forbidden_outputs": ["type-name phrases", "original surface leaks", "direct identifiers"],
    }
    packet["artifact_slice_hashes"] = {"nearby_profile_rows": _hash_payload(relevant)}
    packet["context_packet_hash"] = _hash_payload(packet)
    return packet


def ensure_local_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"llama-swap base URL must be local, got {base_url}")


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / "lattice_producer" / f"{key}.json"


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
) -> dict[str, Any]:
    ensure_local_base_url(base_url)
    packet = assemble_context_packet(
        item,
        profiles_path=profiles_path,
        run_dir=run_dir,
        prompt_version=prompt_version,
        max_context_rows=max_context_rows,
    )
    identity = {
        "model": model,
        "base_url": base_url,
        "prompt_version": prompt_version,
        "item_id": item.get("item_id"),
        "context_packet_hash": packet["context_packet_hash"],
    }
    cache_key = _hash_payload(identity)
    cache_dir = Path(os.environ.get("INFERDPT_LLM_CACHE", "data/llm_cache"))
    cache_file = _cache_path(cache_dir, cache_key)
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"))
    prompt = (
        "Return strict JSON only. Propose lattice generalization levels for this item. "
        "Do not provide certifying counts.\n\n"
        + json.dumps(packet, sort_keys=True)
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        if not escalation_model:
            payload = {"candidates": [], "parse_error": "invalid_json", "raw": content}
        else:
            response = client.chat.completions.create(
                model=escalation_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
    payload["cache_key"] = cache_key
    payload["context_packet_hash"] = packet["context_packet_hash"]
    atomic_write_json(cache_file, payload)
    return payload
