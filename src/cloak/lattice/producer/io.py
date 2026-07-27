"""Small file helpers for idempotent producer artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl_unique(path: str | Path, rows: Iterable[dict[str, Any]], key: str = "item_id") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {str(row.get(key)) for row in read_jsonl(path) if row.get(key) is not None}
    added = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            row_key = row.get(key)
            if row_key is not None and str(row_key) in seen:
                continue
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            if row_key is not None:
                seen.add(str(row_key))
            added += 1
        fh.flush()
        os.fsync(fh.fileno())
    return added


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
