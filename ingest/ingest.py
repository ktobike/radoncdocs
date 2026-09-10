# ingest/ingest.py
from __future__ import annotations
import argparse, shutil, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import pymupdf as fitz
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import DATA_DIR, CHROMA_DIR, CHUNKS_PATH, COLLECTION, EMBEDDING_MODEL
from demo_metadata import metadata_for_path

# ---------- cleaning helpers ----------
BAD_TOKENS = ["TagedAPTARAEnd","TagedAPTARAP","TagedAPTARAListItem","Download PDF","DOI of original article"]
_hyphen_break = re.compile(r"(\w)-\n(\w)")
_newline_run  = re.compile(r"[ \t]*\n[ \t]*")
_space_run    = re.compile(r"[ \t]{2,}")
_page_num     = re.compile(r"^\s*\d+\s*$")
_demo_page_line = re.compile(r"^\s*Page\s+\d+\s*$", re.I)
_DEMO_BOILERPLATE = {
    "RadOncDocs - DEMO CORPUS",
    "Synthetic documents",
    "Synthetic portfolio demo document. Not clinical, legal, security, or operational guidance.",
}
_POLICY_SECTION_NAMES = (
    "PURPOSE", "SCOPE", "RESPONSIBILITIES", "PROCEDURE", "STEPS", "WORKFLOW",
    "CHECKLIST", "CHECKS", "REQUIREMENTS", "EXCEPTIONS", "RELATED DOCUMENTS",
)
_POLICY_SECTION_RE = re.compile(r"(?m)^\s*(" + "|".join(re.escape(x) for x in _POLICY_SECTION_NAMES) + r")\s*$")

def clean_text(s: str) -> str:
    if not s:
        return ""
    for tok in BAD_TOKENS:
        s = s.replace(tok, " ")
    s = _hyphen_break.sub(r"\1\2", s)
    s = _newline_run.sub("\n", s)
    lines = []
    for line in s.splitlines():
        stripped = line.strip()
        if _page_num.match(line) or _demo_page_line.match(line):
            continue
        if stripped in _DEMO_BOILERPLATE:
            continue
        lines.append(line)
    s = "\n".join(lines)
    s = _space_run.sub(" ", s)
    return s.strip()


def _strip_demo_policy_preamble(text: str, has_manifest_metadata: bool) -> str:
    """Remove title/metadata-table boilerplate from generated demo policy pages.

    The title/category/owner/date stay available as structured metadata, so indexing the
    same fields into page content only creates noisy answers. Keep the actual policy
    sections and any continuation text on later pages.
    """
    if not text or not has_manifest_metadata:
        return text
    match = _POLICY_SECTION_RE.search(text)
    if match:
        return text[match.start():].strip()
    return text.strip()



def _safe_pdf_title(meta: dict, fallback: str) -> str:
    """Prefer useful PDF metadata; ignore generic authoring-tool placeholders."""
    raw = str((meta or {}).get("title") or "").strip()
    if not raw or raw.lower() in {"anonymous", "untitled", "none", "null", "microsoft word"}:
        return fallback
    return raw

def _page_text(page: Any) -> str:
    try:
        return page.get_text("text")
    except Exception:
        if hasattr(page, "getText"):
            return page.getText("text")
        try:
            return page.get_textpage().extractTEXT()
        except Exception:
            return ""

# ---------- policy/page mode ----------
def read_pdf_pages(pdf_path: Path) -> List[Document]:
    docs: List[Document] = []
    with fitz.open(pdf_path) as pdf:
        meta = pdf.metadata or {}  # type: ignore[assignment]
        demo_meta = metadata_for_path(pdf_path)
        title = str(demo_meta.get("title") or _safe_pdf_title(meta, pdf_path.stem))
        for i in range(len(pdf)):            # iterate by index
            page = pdf.load_page(i)          # Page object
            raw = _page_text(page)           # <- robust helper
            text = clean_text(raw)
            text = _strip_demo_policy_preamble(text, bool(demo_meta))
            if len(text) < 40:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "title": title,
                    "page": i + 1,
                    "source": str(pdf_path.resolve()),
                    "file": pdf_path.name,
                    "doc_type": "policy",
                    "category": demo_meta.get("category", "Policy"),
                    "owner": demo_meta.get("owner", ""),
                    "updated": demo_meta.get("updated", ""),
                    "doc_id": demo_meta.get("id", ""),
                    "_raw": raw,             # used by chunk() for headings
                },
            ))
    return docs


# detect headings per-page (for nicer “heading” metadata on policy chunks)
_ALLCAPS_HEAD = re.compile(r"(?m)^[A-Z][A-Z 0-9/\-–&]{3,}$")
_COLON_HEAD   = re.compile(r"(?m)^[A-Z][A-Za-z0-9 /\-–&]{3,}:\s*$")
_NUMBERED     = re.compile(r"(?m)^\s*(\d+[\.\)]|[A-Z][\.\)])\s+[A-Z].{2,}$")

def detect_headings(page_text: str) -> list[tuple[int, str]]:
    heads = []
    for m in _ALLCAPS_HEAD.finditer(page_text): heads.append((m.start(), m.group(0).strip()))
    for m in _COLON_HEAD.finditer(page_text):   heads.append((m.start(), m.group(0).strip(" :")))
    for m in _NUMBERED.finditer(page_text):
        line = m.group(0).strip()
        if len(line) <= 80: heads.append((m.start(), line))
    heads.sort(key=lambda x: x[0])
    seen, out = set(), []
    for pos, h in heads:
        key = re.sub(r"\s+"," ", h).upper()
        if key not in seen:
            seen.add(key); out.append((pos, key))
    return out

def assign_section(text: str, heads: list[tuple[int,str]], char_start: int) -> str:
    cur = ""
    for pos, name in heads:
        if pos <= char_start: cur = name
        else: break
    return cur

def _policy_sections(text: str) -> list[tuple[str, str]]:
    """Return section name/body pairs for policy-style documents."""
    hits = list(_POLICY_SECTION_RE.finditer(text or ""))
    if not hits:
        return []
    out: list[tuple[str, str]] = []
    for i, match in enumerate(hits):
        body_start = match.end()
        body_end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        body = (text[body_start:body_end] or "").strip()
        if body:
            out.append((match.group(1).upper(), body))
    return out


def chunk(docs: List[Document]) -> List[Document]:
    """Chunk policy PDFs by semantic section before falling back to generic splitting.

    Section-aware chunks keep PROCEDURE/RESPONSIBILITIES/REQUIREMENTS content separate,
    which prevents document metadata or neighboring sections from leaking into answers.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=760,
        chunk_overlap=90,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=760,
        chunk_overlap=90,
    )
    out: List[Document] = []
    for d in docs:
        md_base = {k: v for k, v in d.metadata.items() if k != "_raw"}
        sections = _policy_sections(d.page_content)
        if sections:
            for section, body in sections:
                for c_idx, piece in enumerate(splitter.split_text(body)):
                    md = dict(md_base)
                    md["heading"] = section
                    md["section"] = section
                    md["chunk_index"] = c_idx
                    out.append(Document(page_content=piece.strip(), metadata=md))
            continue

        # Continuation pages can start mid-section. They are still useful for retrieval,
        # but keep them isolated from title/metadata boilerplate.
        for c_idx, piece in enumerate(fallback_splitter.split_text(d.page_content)):
            md = dict(md_base)
            md["chunk_index"] = c_idx
            out.append(Document(page_content=piece.strip(), metadata=md))
    return out

# ---------- journal sectionizer mode ----------
_SEC_CANON = {
    "ABSTRACT":"ABSTRACT","BACKGROUND":"BACKGROUND","INTRODUCTION":"INTRODUCTION",
    "MATERIALS AND METHODS":"METHODS","PATIENTS AND METHODS":"METHODS","METHODS":"METHODS",
    "RESULTS":"RESULTS","DISCUSSION":"DISCUSSION","CONCLUSION":"CONCLUSIONS","CONCLUSIONS":"CONCLUSIONS",
    "LIMITATION":"LIMITATIONS","LIMITATIONS":"LIMITATIONS",
    "DATA AVAILABILITY":"DATA AVAILABILITY",
}
_SEC_RE = re.compile(r"(?m)^\s*([A-Z][A-Z0-9 /&\-\(\)]{3,}|[A-Z][a-zA-Z ]{3,}):\s*$")
def _canon(name: str) -> str:
    n = re.sub(r"\s+"," ", (name or "").strip().upper())
    return _SEC_CANON.get(n, n)

def _extract_sections_from_pages(pages: list[str]) -> list[tuple[str,int,int,str]]:
    sep = "\n\n\x0c\n\n"
    joined = sep.join(pages)
    page_starts, ofs = [], 0
    for p in pages:
        page_starts.append(ofs); ofs += len(p) + len(sep)
    hits = list(_SEC_RE.finditer(joined))
    if not hits: return [("FULLTEXT", 1, len(pages), joined.strip())]

    def char_to_page(ci: int) -> int:
        pg = 1
        for i, st in enumerate(page_starts):
            if st <= ci: pg = i + 1
            else: break
        return pg

    out: list[tuple[str,int,int,str]] = []
    for i, m in enumerate(hits):
        start = m.end()
        end   = hits[i+1].start() if i+1 < len(hits) else len(joined)
        name  = _canon(m.group(1))
        blk   = joined[start:end].strip()
        if len(blk) >= 60:
            out.append((name, char_to_page(start), char_to_page(end-1), blk))
    return out

def chunk_journal(pdf_path: Path) -> List[Document]:
    with fitz.open(pdf_path) as pdf:
        meta = pdf.metadata or {}  # type: ignore[assignment]
        demo_meta = metadata_for_path(pdf_path)
        title = str(demo_meta.get("title") or _safe_pdf_title(meta, pdf_path.stem))
        pages = [ clean_text(_page_text(pdf.load_page(i))) for i in range(len(pdf)) ]
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n"," "], chunk_size=900, chunk_overlap=150)
    sections = _extract_sections_from_pages(pages)

    out: List[Document] = []
    for s_idx, (sec, pf, pt, text) in enumerate(sections):
        pieces = splitter.split_text(text)
        cursor = 0
        for c_idx, piece in enumerate(pieces):
            pos = text.find(piece, cursor)
            cursor = (pos if pos >= 0 else cursor) + len(piece)
            out.append(Document(
                page_content=piece,
                metadata={
                    "title": title, "section": sec,
                    "section_index": s_idx, "chunk_index": c_idx,
                    "page_from": pf, "page_to": pt,
                    "source": str(pdf_path.resolve()), "file": pdf_path.name,
                    "doc_type": "journal",
                    "category": demo_meta.get("category", "Research"),
                    "owner": demo_meta.get("owner", ""),
                    "updated": demo_meta.get("updated", ""),
                    "doc_id": demo_meta.get("id", ""),
                }
            ))
    return out

# ---------- embeddings + export ----------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    

def export_jsonl(chunks: List[Document], path: Path = CHUNKS_PATH, append: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for d in chunks:
            json.dump({"text": d.page_content, "metadata": d.metadata}, f, ensure_ascii=False)
            f.write("\n")

def _is_journal_path(p: Path) -> bool:
    return any(part.lower() == "redjournal" for part in p.parts)

def index_single(pdf_path: Path) -> int:
    """Add a single PDF into the existing Chroma index."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return 0
    emb = get_embeddings()
    vs = Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=emb)
    if _is_journal_path(pdf_path):
        docs = chunk_journal(pdf_path)
    else:
        docs = chunk(read_pdf_pages(pdf_path))
    if not docs:
        return 0
    vs.add_documents(docs)
    # append to jsonl
    export_jsonl(docs, path=CHUNKS_PATH, append=True)
    return len(docs)

def delete_single(pdf_path: Path) -> int:
    """Delete all vectors for a single PDF from Chroma using metadata filter."""
    emb = get_embeddings()
    vs = Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=emb)
    try:
        # langchain-chroma supports where filter on metadata
        vs.delete(where={"source": str(pdf_path.resolve())})
        return 1
    except Exception:
        return 0

# ---------- build ----------
def build_index(rebuild: bool = False):
    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(exist_ok=True)

    pdfs = sorted(p for p in DATA_DIR.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR.resolve()}."); return

    print(f"Found {len(pdfs)} PDFs. Parsing…")
    all_chunks: List[Document] = []
    policy_pages: List[Document] = []

    for p in pdfs:
        if any(part.lower() == "redjournal" for part in p.parts):
            all_chunks.extend(chunk_journal(p))
        else:
            policy_pages.extend(read_pdf_pages(p))

    if policy_pages:
        print(f"Chunking {len(policy_pages)} policy pages…")
        all_chunks.extend(chunk(policy_pages))

    print(f"Created {len(all_chunks)} chunks. Embedding + indexing…")
    emb = get_embeddings()
    vs = Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=emb)
    vs.add_documents(all_chunks)
    export_jsonl(all_chunks, append=False)
    print("✅ Index built at", CHROMA_DIR.resolve())
def _open_vs():
    emb = get_embeddings()
    return Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=emb)

def _append_jsonl(chunks: list[Document], path: Path = CHUNKS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for d in chunks:
            json.dump({"text": d.page_content, "metadata": d.metadata}, f, ensure_ascii=False)
            f.write("\n")

def index_pdf(pdf_path: Path) -> Tuple[int, str]:
    """Chunk a single PDF, add to Chroma, and append to chunks.jsonl. Returns (#chunks, doc_type)."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("PDF not found or wrong extension")

    # journal vs policy routing
    is_journal = any(part.lower() == "redjournal" for part in pdf_path.parts)
    if is_journal:
        chunks = chunk_journal(pdf_path)
        doc_type = "journal"
    else:
        pages = read_pdf_pages(pdf_path)
        chunks = chunk(pages)
        doc_type = "policy"

    if not chunks:
        return 0, doc_type

    vs = _open_vs()
    vs.add_documents(chunks)
    _append_jsonl(chunks)
    return len(chunks), doc_type

def delete_pdf_from_index(file_name_or_path: str) -> int:
    """
    Remove vectors for a PDF (by file name) and rewrite chunks.jsonl without those rows.
    Returns #deleted vectors according to vector db (approx).
    """
    p = Path(file_name_or_path)
    fname = p.name
    fname_lower = fname.lower()

    vs = _open_vs()
    # Chroma metadata filters are case-sensitive, so use the stored filename.
    try:
        n = vs.delete(where={"file": {"$eq": fname}})
    except Exception:
        # some Chroma versions return None; we’ll still rewrite JSONL
        n = 0

    # rewrite chunks.jsonl filtering out this file
    jin = CHUNKS_PATH
    if jin.exists():
        rows = []
        with jin.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    meta = obj.get("metadata") or {}
                    if (meta.get("file") or "").lower() != fname_lower:
                        rows.append(line)
                except Exception:
                    continue
        with jin.open("w", encoding="utf-8") as f:
            for line in rows: f.write(line)

    return int(n or 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PDF -> Chroma index")
    ap.add_argument("--rebuild", action="store_true", help="wipe existing index first")
    args = ap.parse_args()
    build_index(rebuild=args.rebuild)
