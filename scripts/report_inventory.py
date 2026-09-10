from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHUNKS = ROOT / "data" / "chunks.jsonl"  # optional; if your ingest wrote this

phrase_list = ["on-call", "on call", "escalation", "pager", "after-hours", "after hours", "coverage"]
phrase_re = re.compile("|".join(re.escape(p) for p in phrase_list), re.IGNORECASE)

def kind_for(p: Path) -> str:
    s = str(p).lower()
    if "policies and procedures" in s:
        return "policy"
    if "redjournal" in s or "journal" in s:
        return "journal"
    return "other"

def list_pdfs(base: Path):
    files = sorted([p for p in base.rglob("*.pdf") if p.is_file()])
    by_kind = defaultdict(list)
    for p in files:
        by_kind[kind_for(p)].append(p)
    return by_kind

def scan_chunks(chunks_path: Path):
    hits_by_file = defaultdict(int)
    total_hits = 0
    if not chunks_path.exists():
        return {"total_hits": 0, "hits_by_file": {}, "has_chunks": False}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or "")[:30000]
            m = phrase_re.findall(text)
            if m:
                src = obj.get("metadata", {}).get("source") or obj.get("source") or "unknown"
                hits_by_file[src] += len(m)
                total_hits += len(m)
    return {"total_hits": total_hits, "hits_by_file": dict(hits_by_file), "has_chunks": True}

def main():
    print("== Document inventory ==")
    if not RAW.exists():
        print(f"Missing folder: {RAW}")
        return
    inv = list_pdfs(RAW)
    for k in ["policy","journal","other"]:
        items = inv.get(k, [])
        print(f"{k:>7}: {len(items)} files")
        # Show top 8 names to eyeball what’s there
        for p in items[:8]:
            print(f"  - {p.relative_to(RAW)}")
        if len(items) > 8:
            print(f"  ... (+{len(items)-8} more)")
    print()

    cov = scan_chunks(CHUNKS)
    if not cov["has_chunks"]:
        print(f"NOTE: {CHUNKS} not found. (Coverage-by-phrase will be skipped.)")
        return

    print("== Keyword coverage in indexed chunks ==")
    print(f"Total matches for {phrase_list}: {cov['total_hits']}")
    if not cov["total_hits"]:
        print("No matches in chunks. Likely those topics aren’t in your current policy PDFs.")
        return

    # Show top 10 files by hits
    top = sorted(cov["hits_by_file"].items(), key=lambda kv: kv[1], reverse=True)[:10]
    for src, n in top:
        lbl = "policy" if "Policies and Procedures" in src else ("journal" if "RedJournal" in src or "Journal" in src else "other")
        print(f"- {n:>3} hits | {lbl:<7} | {src}")

if __name__ == "__main__":
    main()
