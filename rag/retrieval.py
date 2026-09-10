# api/retrieval.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import defaultdict, Counter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from api.utils_device import cuda_or_cpu
from .bm25 import BM25Store
from config import CHROMA_DIR as CONFIG_CHROMA_DIR, CHUNKS_PATH, COLLECTION, EMBEDDING_MODEL, ENABLE_RERANKER


# ==== [BEGIN device picker] ====
def pick_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
# ==== [END device picker] ====



# -------------------------
# Config
# -------------------------
CHROMA_DIR = str(CONFIG_CHROMA_DIR)

# Try optional cross-encoder
try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
except Exception:
    CrossEncoder = None
    _HAS_RERANKER = False

# -------------------------
# Embeddings / Stores
# -------------------------



def _emb():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": cuda_or_cpu()},
        encode_kwargs={"normalize_embeddings": True},
    )



def _vs():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=_emb(),
    )

_bm25: Optional[BM25Store] = None
def _bm():
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Store(str(CHUNKS_PATH))
    return _bm25

# -------------------------
# Doc typing & intent
# -------------------------
DOC_TYPE_PATTERNS = [
    ("policy", re.compile(r"/Policies and Procedures/", re.I)),
    ("journal", re.compile(r"/RedJournal/", re.I)),
]

def doc_type_from_path(path: str) -> str:
    normalized = (path or "").replace("\\", "/")
    for name, pat in DOC_TYPE_PATTERNS:
        if pat.search(normalized):
            return name
    return "other"

PROCEDURE_HINTS = (
    "how to", "how should", "what should i do", "steps", "procedure", "protocol", "workflow",
    "checklist", "checks", "check before", "troubleshoot", "misaligned", "misalignment",
    "scan", "setup", "set up", "acquire", "acquisition", "use the", "configure",
)
POLICY_HINTS = ("policy","sop","guideline","roles","responsibilities","approval","change control")
JOURNAL_HINTS = ("randomized","phase","trial","patients","methods","results","conclusion","ntcp","hazard ratio","let","dosimetry")

def detect_query_intent(q: str) -> str:
    s = (q or "").lower()
    if any(w in s for w in PROCEDURE_HINTS): return "procedure"
    if any(w in s for w in POLICY_HINTS):    return "policy"
    if any(w in s for w in JOURNAL_HINTS):   return "journal"
    return "factoid"

# -------------------------
# Text utils + scoring
# -------------------------
TOKEN = re.compile(r"[A-Za-z0-9]{3,}")
STOP = {
    "the","a","an","and","or","of","for","to","in","on","by","with","at","as","is","are",
    "what","which","who","when","where","how","do","does","did","be","can","could","should",
    "please","tell","me","about","policy","policies","procedure","procedures"
}
_token = re.compile(r"[a-z0-9]+")

def _lex_tokens(s: str):
    return set(_token.findall((s or "").lower()))

def _section_boost(intent: str, heading_or_section: str) -> float:
    h = (heading_or_section or "").lower()
    if intent == "procedure" and any(k in h for k in ("procedure","steps","workflow")):
        return 0.25
    if intent == "journal" and any(k in h for k in ("abstract","results","conclusion","conclusions","discussion")):
        return 0.15
    return 0.0


def _page_prior(page: int) -> float:
    # prefer earlier pages a bit, tapering after ~10 pages
    p = max(1, int(page or 1))
    return max(0.0, 0.18 * (1.0 - (p - 1) / 10.0))

def _keywords(s: str) -> List[str]:
    return [w for w in TOKEN.findall((s or "").lower()) if len(w) >= 3 and w not in STOP]

def _terms_in(s: str, terms: List[str]) -> int:
    t = (s or "").lower()
    return sum(1 for w in set(terms) if w in t)


def _query_section_boost(question: str, section: str) -> float:
    """Boost the section that best matches the user's information need."""
    q = (question or "").lower()
    sec = (section or "").upper()
    if not sec:
        return 0.0

    if any(x in q for x in ("responsib", "who is responsible", "who should")):
        return 0.34 if sec == "RESPONSIBILITIES" else 0.0
    if any(x in q for x in ("what should i do", "how should", "how do i", "steps", "workflow", "check", "checks", "before", "troubleshoot", "misalign")):
        if sec in {"PROCEDURE", "STEPS", "WORKFLOW", "CHECKLIST", "CHECKS"}:
            return 0.32
        if sec == "REQUIREMENTS":
            return 0.08
    if any(x in q for x in ("requirement", "required", "must", "prohibited")):
        if sec == "REQUIREMENTS":
            return 0.28
        if sec == "PROCEDURE":
            return 0.08
    if "scope" in q and sec == "SCOPE":
        return 0.30
    if any(x in q for x in ("purpose", "why does", "why is")) and sec == "PURPOSE":
        return 0.30
    if any(x in q for x in ("exception", "allowed to bypass", "waiver")) and sec == "EXCEPTIONS":
        return 0.30
    return 0.0

def _doc_type(source: str | None, title: str | None, file: str | None) -> str:
    # prefer path-based detection (stable), fall back to title/file
    source_norm = (source or "").replace("\\", "/").lower()
    combined = " ".join([source_norm, title or "", file or ""]).lower()
    if "/policies and procedures/" in source_norm or any(x in combined for x in (" policy"," procedure"," sop"," guideline","handbook")):
        return "policy"
    if "/redjournal/" in source_norm or "journal" in combined:
        return "journal"
    return "unknown"

def _intent(query: str) -> str:
    q = (query or "").lower()
    policy_hints = ("policy","procedure","sop","approve","approval","form","workflow","who","steps","how do i")
    journal_hints = ("study","paper","trial","cohort","results","conclusion","outcomes","figure","dose-let")
    if any(h in q for h in policy_hints):  return "policy"
    if any(h in q for h in journal_hints): return "journal"
    return "generic"

def score_chunk(
    question: str,
    ch: dict,
    intent: str,
    major_doctype: Optional[str] = None,
) -> float:
    base = float(ch.get("score", 0.0))
    qtok = _lex_tokens(question)
    ctok = _lex_tokens(ch.get("text",""))
    overlap = len(qtok & ctok) / (1 + len(qtok))  
    sboost = _section_boost(intent, ch.get("section","") or ch.get("heading",""))
    pprior = _page_prior(ch.get("page", 1))
    coboost = 0.15 if major_doctype and ch.get("doc_type") == major_doctype else 0.0
    return base + overlap + sboost + pprior + coboost

# -------------------------
# Optional reranker
# -------------------------
from typing import Any, Optional, cast

_RERANKER: Optional[Any] = None


def _get_reranker():
    global _RERANKER
    if ENABLE_RERANKER and _RERANKER is None and CrossEncoder is not None:
        dev = "cuda" if cuda_or_cpu() == "cuda" else "cpu"
        _RERANKER = CrossEncoder(
            "BAAI/bge-reranker-large",
            device=dev,
            trust_remote_code=True
        )
    return _RERANKER
from collections import defaultdict as _dd
from collections import defaultdict

def _expand_by_section(
    finals: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    max_neighbors: int = 3,
) -> List[Dict[str, Any]]:
    """
    For journal chunks that include section metadata, add up to `max_neighbors`
    neighboring chunks from the *same (file, section)* around each selected hit.
    This keeps RESULTS/CONCLUSIONS sentences together for reranking & answering.
    If no section metadata is present (e.g., policy docs), returns `finals` unchanged.
    """
    if not finals:
        return finals

    # Only run if at least one selected hit has a section
    if not any(h.get("section") for h in finals):
        return finals

    # Index the full candidate pool by (file, section)
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for h in pool:
        key = ((h.get("file") or "").lower(), (h.get("section") or "").lower())
        index[key].append(h)

    out: List[Dict[str, Any]] = []
    seen: set = set()

    for h in finals:
        key = ((h.get("file") or "").lower(), (h.get("section") or "").lower())
        bundle = index.get(key, [h])

        # Sort deterministically inside a section
        bundle = sorted(
            bundle,
            key=lambda x: (
                int(x.get("section_index") or 0),
                int(x.get("chunk_index") or 0),
            ),
        )

        # Find the position of the chosen hit within its section bundle
        try:
            pos = next(
                i for i, b in enumerate(bundle)
                if (b.get("section_index") == h.get("section_index")
                    and b.get("chunk_index") == h.get("chunk_index"))
            )
        except StopIteration:
            pos = 0  # fallback if metadata is missing

        # Window around the chosen hit
        start = max(0, pos - max_neighbors)
        end = min(len(bundle), pos + max_neighbors + 1)

        for b in bundle[start:end]:
            sig = (b.get("file"), b.get("section_index"), b.get("chunk_index"))
            if sig not in seen:
                seen.add(sig)
                out.append(b)

    return out or finals

# -------------------------
# Retrieval (COMPLETE)
# -------------------------
def retrieve(query: str, k: int = 8, allowed_doc_types: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval (dense + BM25) with file-first routing, intent boosts,
    and optional cross-encoder rerank. Returns up to k page-level hits.
    """
    vs = _vs()
    terms = _keywords(query)
    want = _intent(query)

    # --- dense ---
  
    dense: List[Dict[str, Any]] = []
    try:
        pairs: List[Tuple[Any, float]] = vs.similarity_search_with_relevance_scores(query, k=max(k, 40))
        for doc, rel in pairs:
            md = doc.metadata or {}
            dense.append({
                **md,
                "text": doc.page_content.strip(),
                "dense": float(rel),
                "bm25_norm": 0.0,
            })
    except Exception:
        pairs2: List[Tuple[Any, float]] = vs.similarity_search_with_score(query, k=max(k, 40))
        for doc, dist in pairs2:
            sim = max(0.0, 1.0 - float(dist))
            md = doc.metadata or {}
            dense.append({
                **md,
                "text": doc.page_content.strip(),
                "dense": float(sim),
                "bm25_norm": 0.0,
            })



    # --- bm25 ---
    bm_hits = _bm().search(query, k=max(k, 80))

    # --- merge dense and lexical hits without collapsing distinct sections ---

    def _merge_key(h: Dict[str, Any]) -> tuple:
        file_key = (h.get("file") or "").lower()
        page_key = int(h.get("page") or h.get("page_from") or 0)
        section_key = str(h.get("section") or h.get("heading") or "").lower()
        chunk_key = int(h.get("chunk_index") or 0)
        return (file_key, page_key, section_key, chunk_key)

    merged: Dict[tuple, Dict[str, Any]] = {}
    for h in dense:
        key = _merge_key(h)
        merged[key] = {**h}
    for h in bm_hits:
        key = _merge_key(h)
        if key in merged:
            merged[key]["bm25_norm"] = max(
                float(merged[key].get("bm25_norm", 0.0)),
                float(h.get("bm25_norm", 0.0))
            )
            if len(h.get("text","")) > len(merged[key].get("text","")):
                merged[key]["text"] = h["text"]
            if "heading" in h and not merged[key].get("heading"):
                merged[key]["heading"] = h.get("heading","")
        else:
            merged[key] = {**h, "dense": 0.0}
            if "heading" not in merged[key]:
                merged[key]["heading"] = h.get("heading","")


    # --- base score + intent boosts + coverage gates ---
    pages: List[Dict[str, Any]] = []
    for v in merged.values():
        src = v.get("source") or ""
        title = v.get("title") or ""
        file = v.get("file") or ""
        dtype = _doc_type(src, title, file)
        if allowed_doc_types is not None and dtype not in allowed_doc_types:
            continue

        # coverage: must match >=2 distinct terms overall; at least 1 in filename or title
        overall = _terms_in(" ".join([v.get("text") or "", title, file]), terms)
        namehit = _terms_in(" ".join([title, file]), terms) >= 1 if terms else True
        if terms and (overall < 2 or not namehit):
            continue

        base = 0.60 * float(v.get("dense", 0.0)) + 0.40 * float(v.get("bm25_norm", 0.0))
        boost = 0.0
        if want == "policy" and dtype == "policy":   boost += 0.25
        if want == "journal" and dtype == "journal": boost += 0.15
        boost += _query_section_boost(query, str(v.get("section") or v.get("heading") or ""))

        v["score"] = round(base + boost, 4)
        v["dtype"] = dtype
        pages.append(v)

    if not pages:
        return []

    # --- file-first routing ---
    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pages: by_file[(p.get("file") or "").lower()].append(p)

    file_rows: List[Tuple[str, float]] = []
    for fname, plist in by_file.items():
        max_page = max(pp["score"] for pp in plist)
        bonus = 0.02 * _terms_in(fname, terms)
        file_rows.append((fname, max_page + bonus))
    file_rows.sort(key=lambda x: x[1], reverse=True)

    top_file_count = min(max(1, (k // 4) + 1), 3)
    chosen_files = {f for f, _ in file_rows[:top_file_count]}

    finals: List[Dict[str, Any]] = []
    for f in chosen_files:
        ps = sorted(by_file[f], key=lambda x: x["score"], reverse=True)[:15]
        finals.extend(ps)

    if not finals:
        return []


    finals = _expand_by_section(finals, pages, max_neighbors=3)

  
    reranker = _get_reranker()


    has_journal = any((h.get("dtype") or h.get("doc_type")) == "journal" for h in finals)
    if reranker is not None and (has_journal or _HAS_RERANKER):
        ce_inputs: List[Tuple[str, str]] = [(query, h.get("text") or "") for h in finals]
        try:
            scores = reranker.predict(ce_inputs).tolist()
            for h, s in zip(finals, scores):
                h["rerank"] = float(s)
            finals.sort(key=lambda x: x.get("rerank", 0.0), reverse=True)
        except Exception:
            finals.sort(key=lambda x: x["score"], reverse=True)
    else:
        finals.sort(key=lambda x: x["score"], reverse=True)

    # --- expand by section neighbors to avoid cutting sentences ---
    finals = _expand_by_section(finals, pages, max_neighbors=3)



    # --- dedup + trim ---
    out: List[Dict[str, Any]] = []
    seen = set()
    for h in finals:
        key = (
            h.get("file") or "",
            int(h.get("page") or h.get("page_from") or 0),
            str(h.get("section") or h.get("heading") or ""),
            int(h.get("chunk_index") or 0),
        )
        if key in seen: continue
        seen.add(key)
        h["score"] = float(f"{float(h.get('score', 0.0)):.3f}")
        out.append(h)
        if len(out) >= k: break

    return out

# -------------------------
# Context-aware wrapper (call this from /answer)
# -------------------------
def search_with_context(
    question: str,
    k: int = 8,
    min_score: float = 0.0,
    policy_only: bool = False,
    allowed_doc_types: Optional[set[str]] = None):
    """
    Wraps retrieve() with doc-type gating and a score floor.
    Returns: (hits: List[dict], intent: str)
    """
    intent = detect_query_intent(question)

    # Decide the collection before file-first routing. Gating only after retrieval can
    # let strong journal matches occupy every candidate slot for a policy-only query.
    if allowed_doc_types is None:
        if policy_only or intent == "procedure":
            allowed_doc_types = {"policy"}
        elif intent == "journal":
            allowed_doc_types = {"journal"}
        else:
            allowed_doc_types = {"policy", "journal"}
    else:
        allowed_doc_types = set(allowed_doc_types)

    raw = retrieve(question, k=max(k, 12), allowed_doc_types=allowed_doc_types)
    if not raw:
        return [], intent

    # ensure doc_type
    for h in raw:
        if "doc_type" not in h:
            h["doc_type"] = doc_type_from_path(h.get("source") or h.get("file") or "")

    gated = [h for h in raw if h.get("doc_type") in allowed_doc_types]
    # score floor on current score
    gated = [h for h in gated if float(h.get("score") or 0.0) >= float(min_score or 0.0)]

    # cap to k
    out = gated[:k]
    return out, intent

# -------------------------
# Procedure helpers (moved from qa_routes to avoid drift)
# -------------------------
def all_chunks_for_file(file_or_source: str) -> List[Dict[str, Any]]:
    """Return every indexed chunk for one document from the local BM25 mirror.

    This is used after retrieval has already selected the best document. It lets
    procedural answers inspect the document's full PROCEDURE/CHECKLIST section
    instead of depending on whichever single chunk happened to rank first.
    """
    target = str(file_or_source or "").replace("\\", "/").lower()
    if not target:
        return []
    store = _bm()
    out: List[Dict[str, Any]] = []
    for text, md in zip(store.docs, store.metas):
        md = md or {}
        file_val = str(md.get("file") or "").replace("\\", "/").lower()
        source_val = str(md.get("source") or "").replace("\\", "/").lower()
        if target in {file_val, source_val} or target.endswith("/" + file_val) or source_val.endswith("/" + target):
            out.append({**md, "text": str(text).strip(), "score": 0.0})
    out.sort(key=lambda h: (int(h.get("page") or h.get("page_from") or 0), int(h.get("chunk_index") or 0)))
    return out


def q_suggests_procedure(q: str) -> bool:
    """Check if question suggests a procedural/how-to answer."""
    ql = (q or "").lower()
    return any(w in ql for w in (
        "how to", "how should", "what should i do", "steps", "procedure", "workflow",
        "checklist", "checks", "check before", "set up", "setup", "acquire", "acquisition",
        "configure", "scan", "troubleshoot", "misaligned", "misalignment",
    ))


def gather_procedure_steps(hits: List[Dict], want: int = 8) -> List[Tuple[str, Optional[int]]]:
    """Extract bullet/numbered lines from candidate pages."""
    steps: List[Tuple[str, Optional[int]]] = []
    seen = set()
    bullet = re.compile(r"^\s*(?:\d+[\.\)]|[A-Za-z][\.\)]|[-•▪‣*·◦●∙]|[oO])\s+(.*)$")
    
    for h in hits[:6]:  # keep it tight
        page = h.get("page")
        for line in (h.get("text") or "").splitlines():
            m = bullet.match(line)
            if not m:
                continue
            step = re.sub(r"\s*\(see.*?\)$", "", m.group(1)).strip()
            # keep imperative-ish short lines
            if 5 <= len(step) <= 220 and re.match(
                r"^(click|enter|select|verify|check|ensure|place|position|start|stop|record|scan|set|open|type|confirm|submit|obtain|enroll|use|report|create|run|compare|flag|generate|resolve|move|review|correct|freeze|re-run|approve|reject|contact|grant|remove|activate|validate|capture|publish|attach|configure|mark|apply|describe|load|score|time-limit|do not)\b",
                step,
                re.I,
            ):
                key = step.lower()
                if key not in seen:
                    seen.add(key)
                    steps.append((step, page if isinstance(page, int) else None))
                    if len(steps) >= want:
                        return steps
    return steps




def format_steps_with_cites(steps: List[Tuple[str, Optional[int]]], hits: List[Dict]) -> str:
    """Format numbered steps with per-step [n] cites and a compact Sources list.
       Falls back gracefully if section or page range is missing."""
    if not steps:
        return ""

    # Build page -> best source map (use first 10 hits; map all pages in a section range)
    page_map: Dict[int, Tuple[int, Optional[str], Optional[int], Optional[int]]] = {}
    for idx, h in enumerate(hits[:10], 1):
        pf = h.get("page_from") or h.get("page")
        pt = h.get("page_to") or h.get("page")
        sec = h.get("section")
        if isinstance(pf, int) and isinstance(pt, int) and pf > 0 and pt >= pf:
            for p in range(pf, pt + 1):
                if p not in page_map:
                    page_map[p] = (idx, sec, pf, pt)
        elif isinstance(pf, int) and pf > 0:
            page_map.setdefault(pf, (idx, sec, pf, pf))

    def _cite_for_page(pg: Optional[int]) -> str:
        if pg is None:
            return ""
        info = page_map.get(int(pg))
        if not info:
            return f" (p.{pg})"
        idx, sec, pf, pt = info
        if isinstance(pf, int) and isinstance(pt, int):
            pg_str = f"pp.{pf}-{pt}" if pt != pf else f"p.{pf}"
        else:
            pg_str = f"p.{pg}"
        sec_str = f"{sec}, " if sec else ""
        return f" [{idx}] ({sec_str}{pg_str})"

    lines = [f"{i}) {s}{_cite_for_page(pg)}" for i, (s, pg) in enumerate(steps, 1)]

    # Detailed Sources list (first 6)
    src_lines: List[str] = []
    max_src = min(6, len(hits))
    for i, h in enumerate(hits[:max_src], 1):
        title = h.get("title") or h.get("file") or "(untitled)"
        sec = h.get("section")
        pf = h.get("page_from") or h.get("page")
        pt = h.get("page_to") or h.get("page")
        if isinstance(pf, int) and isinstance(pt, int):
            pg_str = f"pp.{pf}-{pt}" if pt != pf else f"p.{pf}"
        elif isinstance(pf, int):
            pg_str = f"p.{pf}"
        else:
            pg_str = ""
        sec_part = f"{sec}: " if sec else ""
        tail = f"{sec_part}{pg_str}".rstrip()
        src_lines.append(f"[{i}] {title}" + (f" — {tail}" if tail else ""))

    return "\n".join(lines) + ("\n\nSources:\n" + "\n".join(src_lines) if src_lines else "")


