"""Small JSON utilities for the standalone entry points."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path, *, unique_contexts=False):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected a nonempty JSONL file of objects")
    ids = [row["context_id"] for row in rows if "context_id" in row]
    if unique_contexts and len(ids) != len(set(ids)):
        raise ValueError("Duplicate context_id")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
