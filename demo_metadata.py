from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from config import DATA_DIR

MANIFEST_PATH = DATA_DIR / "manifest.json"


def _norm(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


@lru_cache(maxsize=1)
def manifest_rows() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


@lru_cache(maxsize=1)
def manifest_by_rel() -> dict[str, dict]:
    return {_norm(str(row.get("rel") or "")): row for row in manifest_rows() if row.get("rel")}


@lru_cache(maxsize=1)
def manifest_by_file() -> dict[str, dict]:
    return {str(row.get("file") or "").lower(): row for row in manifest_rows() if row.get("file")}


def metadata_for_path(path: Path) -> dict:
    try:
        rel = _norm(str(path.resolve().relative_to(DATA_DIR.resolve())))
    except Exception:
        rel = _norm(str(path))
    row = manifest_by_rel().get(rel)
    if row:
        return dict(row)
    return dict(manifest_by_file().get(path.name.lower(), {}))
