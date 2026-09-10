from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RADONCDOCS_DATA_DIR", str(ROOT_DIR / "data" / "demo"))).expanduser().resolve()
CHROMA_DIR = ROOT_DIR / "chroma"
CHUNKS_PATH = ROOT_DIR / "data" / "chunks.jsonl"
SITE_DIR = ROOT_DIR / "site"
CACHE_DIR = ROOT_DIR / ".cache" / "txt"
UI_DIR = ROOT_DIR / "ui"

COLLECTION = os.getenv("RADONCDOCS_COLLECTION", "radoncdocs_demo")
EMBEDDING_MODEL = os.getenv("RADONCDOCS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
ENABLE_RERANKER = os.getenv("RADONCDOCS_ENABLE_RERANKER", "0") == "1"
ENABLE_ADMIN = os.getenv("RADONCDOCS_ENABLE_ADMIN", "0") == "1"
ENABLE_WRITER = os.getenv("RADONCDOCS_ENABLE_WRITER", "0") == "1"
SERVE_SOURCE_FILES = os.getenv("RADONCDOCS_SERVE_SOURCE_FILES", "1") == "1"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SITE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
