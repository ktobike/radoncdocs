from __future__ import annotations
from typing import List, Dict, Any
import json, re
from pathlib import Path
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[A-Za-z0-9]{3,}")

class BM25Store:
    def __init__(self, jsonl_path: str):
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Index metadata not found: {path}. Run the ingest rebuild first.")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.docs.append(obj["text"])
                self.metas.append(obj["metadata"])
        self.tokens = [self._tok(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokens)

    def _tok(self, s: str) -> List[str]:
        return TOKEN.findall(s.lower())

    def search(self, query: str, k: int = 25) -> List[Dict[str, Any]]:
        q = self._tok(query)
        scores = self.bm25.get_scores(q)
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        top = max((scores[i] for i in idxs), default=1.0)
        for i in idxs:
            md = self.metas[i] or {}
            out.append({
                **md,
                "text": self.docs[i].strip(),
                "bm25": float(scores[i]),
                "bm25_norm": float(scores[i] / top if top > 0 else 0.0),
            })
        return out
