"""Download public candidate benchmark datasets into data/external.

This is an explicit data-acquisition script, not runtime code. It records source, license, revision,
and downloaded files in a per-dataset MANIFEST.json so later corpus builders can depend on local
artifacts without re-contacting public services.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RAW_DIR = Path("data/external")
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
USER_AGENT = "agent-cloak-benchmark-dataset-fetcher/1.0"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    provider: str
    license: str
    homepage: str
    note: str
    rel_dir: str
    repo_id: str | None = None
    revision: str = "main"
    archive_url: str | None = None
    mendeley_id: str | None = None
    mendeley_version: int | None = None


@dataclass(frozen=True)
class FileSpec:
    path: str
    url: str
    size: int | None = None
    sha256: str | None = None


SOURCES: dict[str, DatasetSource] = {
    "primock57": DatasetSource(
        name="primock57",
        provider="github-archive",
        license="CC BY 4.0",
        homepage="https://github.com/babylonhealth/primock57",
        note=(
            "Downloads the GitHub source archive. Git LFS audio blobs may remain pointers; "
            "transcripts, notes, README, and license are included in the archive."
        ),
        rel_dir="primock57",
        archive_url="https://codeload.github.com/babylonhealth/primock57/zip/refs/heads/main",
        revision="main",
    ),
    "rat-bench": DatasetSource(
        name="rat-bench",
        provider="huggingface",
        license="MIT",
        homepage="https://huggingface.co/datasets/imperial-cpg/rat-bench",
        note="Synthetic direct/indirect identifier benchmark for text anonymization.",
        rel_dir="rat-bench",
        repo_id="imperial-cpg/rat-bench",
        revision="main",
    ),
    "synthetic-financial-pii": DatasetSource(
        name="synthetic-financial-pii",
        provider="mendeley",
        license="CC BY 4.0",
        homepage="https://data.mendeley.com/datasets/tzrjx692jy/1",
        note="Synthetic financial-document PII detection/anonymization data.",
        rel_dir="synthetic-financial-pii",
        mendeley_id="tzrjx692jy",
        mendeley_version=1,
    ),
    "pii-bench": DatasetSource(
        name="pii-bench",
        provider="huggingface",
        license="Apache-2.0",
        homepage="https://huggingface.co/datasets/Pritesh-2711/pii-bench",
        note=(
            "Large PIIBench token-classification corpus. Use --include or --max-bytes 0 "
            "intentionally; train.jsonl is over 1 GB."
        ),
        rel_dir="pii-bench",
        repo_id="Pritesh-2711/pii-bench",
        revision="main",
    ),
}


def source_report() -> dict[str, Any]:
    return {
        "default_raw_dir": str(DEFAULT_RAW_DIR),
        "default_max_bytes": DEFAULT_MAX_BYTES,
        "sources": {
            name: {
                "provider": src.provider,
                "homepage": src.homepage,
                "license": src.license,
                "note": src.note,
                "repo_id": src.repo_id,
                "revision": src.revision,
                "rel_dir": src.rel_dir,
            }
            for name, src in sorted(SOURCES.items())
        },
    }


def _request(url: str, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(_request(url, "application/json"), timeout=60) as resp:
        return json.load(resp)


def _url_join_hf(repo_id: str, revision: str, path: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted}"


def hf_file_specs(repo_id: str, revision: str, tree_rows: Iterable[dict[str, Any]]) -> list[FileSpec]:
    specs = []
    for row in tree_rows:
        if row.get("type") != "file":
            continue
        path = row["path"]
        specs.append(FileSpec(path=path, url=_url_join_hf(repo_id, revision, path), size=row.get("size")))
    return specs


def discover_hf_files(src: DatasetSource) -> tuple[list[FileSpec], dict[str, Any]]:
    if not src.repo_id:
        raise ValueError(f"{src.name} has no Hugging Face repo id")
    api_base = f"https://huggingface.co/api/datasets/{src.repo_id}"
    info = _get_json(api_base)
    revision = info.get("sha") or src.revision
    tree_url = f"{api_base}/tree/{src.revision}?recursive=1"
    tree = _get_json(tree_url)
    meta = {
        "repo_id": src.repo_id,
        "revision": revision,
        "last_modified": info.get("lastModified"),
        "used_storage": info.get("usedStorage"),
        "tags": info.get("tags", []),
    }
    return hf_file_specs(src.repo_id, src.revision, tree), meta


def mendeley_file_specs(rows: Iterable[dict[str, Any]]) -> list[FileSpec]:
    specs = []
    for row in rows:
        details = row.get("content_details", {})
        url = details.get("download_url")
        if not url:
            continue
        specs.append(
            FileSpec(
                path=row["filename"],
                url=url,
                size=row.get("size") or details.get("size"),
                sha256=details.get("sha256_hash"),
            )
        )
    return specs


def discover_mendeley_files(src: DatasetSource) -> tuple[list[FileSpec], dict[str, Any]]:
    if not src.mendeley_id or not src.mendeley_version:
        raise ValueError(f"{src.name} has no Mendeley id/version")
    url = (
        f"https://data.mendeley.com/public-api/datasets/{src.mendeley_id}/files"
        f"?folder_id=root&version={src.mendeley_version}"
    )
    rows = _get_json(url)
    meta = {"mendeley_id": src.mendeley_id, "version": src.mendeley_version, "files_api": url}
    return mendeley_file_specs(rows), meta


def discover_github_archive(src: DatasetSource) -> tuple[list[FileSpec], dict[str, Any]]:
    if not src.archive_url:
        raise ValueError(f"{src.name} has no archive URL")
    filename = f"{src.rel_dir}-{src.revision}.zip"
    meta = {"archive_url": src.archive_url, "revision": src.revision}
    return [FileSpec(path=filename, url=src.archive_url, size=None)], meta


def discover_files(src: DatasetSource) -> tuple[list[FileSpec], dict[str, Any]]:
    if src.provider == "huggingface":
        return discover_hf_files(src)
    if src.provider == "mendeley":
        return discover_mendeley_files(src)
    if src.provider == "github-archive":
        return discover_github_archive(src)
    raise ValueError(f"unsupported provider for {src.name}: {src.provider}")


def apply_includes(files: list[FileSpec], includes: list[str]) -> list[FileSpec]:
    if not includes:
        return files
    return [f for f in files if any(fnmatch.fnmatch(f.path, pat) for pat in includes)]


def known_size(files: Iterable[FileSpec]) -> int:
    return sum(f.size or 0 for f in files)


def enforce_size_budget(dataset: str, files: list[FileSpec], max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    total = known_size(files)
    if total > max_bytes:
        pretty_total = _format_bytes(total)
        pretty_limit = _format_bytes(max_bytes)
        raise SystemExit(
            f"{dataset} selected files are {pretty_total}, above --max-bytes {pretty_limit}. "
            "Use --include to select a smaller subset or --max-bytes 0 to disable this guard."
        )


def _format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(spec: FileSpec, out: Path, *, dry_run: bool = False, skip_existing: bool = True) -> dict[str, Any]:
    print(f"fetch {spec.url} -> {out}", flush=True)
    if dry_run:
        return {"path": spec.path, "size": spec.size, "sha256": spec.sha256, "status": "dry-run"}
    if skip_existing and out.exists():
        if spec.sha256 and _sha256(out) != spec.sha256:
            raise SystemExit(f"existing file checksum mismatch: {out}")
        return {"path": spec.path, "size": out.stat().st_size, "sha256": spec.sha256, "status": "exists"}

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(_request(spec.url), timeout=60) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1024 * 1024)
        if spec.sha256:
            got = _sha256(tmp)
            if got != spec.sha256:
                raise SystemExit(f"checksum mismatch for {spec.path}: expected {spec.sha256}, got {got}")
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": spec.path, "size": out.stat().st_size, "sha256": spec.sha256, "status": "downloaded"}


def write_manifest(
    dataset_dir: Path,
    src: DatasetSource,
    provider_meta: dict[str, Any],
    files: list[FileSpec],
    results: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> Path:
    manifest = {
        "schema_version": 1,
        "created": datetime.now(timezone.utc).isoformat(),
        "dataset": src.name,
        "provider": src.provider,
        "homepage": src.homepage,
        "license": src.license,
        "note": src.note,
        "source": provider_meta,
        "dry_run": dry_run,
        "selected_size": known_size(files),
        "selected_files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in files],
        "results": results,
    }
    path = dataset_dir / "MANIFEST.json"
    if not dry_run:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def fetch_dataset(
    name: str,
    raw_dir: Path,
    *,
    includes: list[str],
    dry_run: bool,
    max_bytes: int,
    skip_existing: bool,
) -> dict[str, Any]:
    src = SOURCES[name]
    files, provider_meta = discover_files(src)
    files = apply_includes(files, includes)
    if not files:
        raise SystemExit(f"{name}: no files selected")
    enforce_size_budget(name, files, max_bytes)

    dataset_dir = raw_dir / src.rel_dir
    results = []
    print(
        f"{name}: {len(files)} files, known size {_format_bytes(known_size(files))}, license {src.license}",
        flush=True,
    )
    for spec in files:
        results.append(download_file(spec, dataset_dir / spec.path, dry_run=dry_run, skip_existing=skip_existing))
    manifest_path = write_manifest(dataset_dir, src, provider_meta, files, results, dry_run=dry_run)
    if dry_run:
        print(f"{name}: dry-run only; manifest would be {manifest_path}", flush=True)
    else:
        print(f"{name}: wrote manifest {manifest_path}", flush=True)
    return {"dataset": name, "files": len(files), "known_size": known_size(files), "manifest": str(manifest_path)}


def selected_datasets(dataset_args: list[str] | None, all_sources: bool) -> list[str]:
    selected = list(dataset_args or [])
    if all_sources:
        selected.extend(SOURCES)
    if not selected:
        raise SystemExit("choose --dataset, --all, or --list")
    return list(dict.fromkeys(selected))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--dataset", action="append", choices=sorted(SOURCES))
    ap.add_argument("--all", action="store_true", help="select all supported datasets")
    ap.add_argument("--include", action="append", default=[], help="glob path to include, repeatable")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                    help="fail if selected known bytes exceed this; use 0 for no limit")
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(source_report(), indent=2, sort_keys=True), flush=True)
        return

    summaries = []
    for dataset in selected_datasets(args.dataset, args.all):
        try:
            summaries.append(
                fetch_dataset(
                    dataset,
                    args.raw_dir,
                    includes=args.include,
                    dry_run=args.dry_run,
                    max_bytes=args.max_bytes,
                    skip_existing=not args.no_skip_existing,
                )
            )
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"{dataset}: HTTP {exc.code} while fetching {exc.url}") from exc
    print(json.dumps({"downloaded": summaries}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
