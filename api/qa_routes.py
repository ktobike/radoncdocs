# pyright: reportMissingImports=false
# api/qa_routes.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
from urllib.parse import quote
import re
from fastapi import APIRouter
from pydantic import BaseModel

from rag.retrieval import search_with_context, q_suggests_procedure, gather_procedure_steps, format_steps_with_cites, all_chunks_for_file
from config import ENABLE_WRITER, DATA_DIR, SERVE_SOURCE_FILES

router = APIRouter(prefix="", tags=["qa"])

class Ask(BaseModel):
    question: str
    policy_only: bool = True
    k: int = 8
    min_score: float | None = 0.25
    allowed_paths: Optional[List[str]] = None

STOP = {
    "the","a","an","and","or","of","for","to","in","on","by","with","at","as","is","are",
    "what","which","who","when","where","how","do","does","did","be","can","could","should",
    "please","tell","me","about","policy","policies","procedure","procedures"
}
_token = re.compile(r"[a-z0-9]+")

def extract_terms(q: str) -> List[str]:
    w = _token.findall((q or "").lower()); return [t for t in w if t not in STOP and len(t) >= 3]
def term_hits(s: str, terms: List[str]) -> int:
    low = (s or "").lower(); return sum(1 for t in set(terms) if t in low)
def _is_allowed(hit: Dict, allow: List[str]) -> bool:
    blob = " ".join([str(hit.get("source") or ""), str(hit.get("title") or ""), str(hit.get("file") or "")]).lower()
    return any(pat.lower() in blob for pat in (allow or []))
def _lex_tokens(s: str) -> set[str]:
    return set(_token.findall((s or "").lower()))
def _looks_relevant(hit: Dict, question: str, min_cov: float = 0.15) -> bool:
    # Section-aware chunks often omit the document title by design, so judge
    # relevance against content *and* structured metadata rather than body text alone.
    qtok = {_stem_token(t) for t in extract_terms(question)}
    blob = " ".join([
        str(hit.get("text") or ""),
        str(hit.get("title") or ""),
        str(hit.get("file") or ""),
        str(hit.get("section") or hit.get("heading") or ""),
    ])
    ctok = {_stem_token(t) for t in _token.findall(blob.lower())}
    if not qtok or not ctok:
        return True
    cov = len(qtok & ctok) / max(1, len(qtok))
    return cov >= min_cov


DOC_KIND_WORDS = {"policy", "procedure", "sop", "standard", "runbook", "checklist", "guideline", "guidelines", "matrix", "guide"}
TITLE_GENERIC = {"research", "document", "documents", "current", "demo", "users", "user", "responsible", "responsibilities", "required", "requirements"}

def _focus_explicit_document(question: str, hits: List[Dict]) -> List[Dict]:
    """Keep an explicitly named policy/procedure from being diluted by related docs.

    General workflow questions can legitimately synthesize multiple sources. But when a
    user asks about *the remote access policy* or names another document-like artifact,
    a strong title match should dominate the answer.
    """
    if len(hits) < 2:
        return hits
    q_tokens = {_stem_token(t) for t in _token.findall((question or "").lower())}
    explicit = bool(q_tokens & DOC_KIND_WORDS) or bool(re.search(r"\bunder\s+(?:the\s+)?[^?.]{2,80}\b(policy|procedure|sop|standard|runbook|checklist|guideline|matrix|guide)\b", question, re.I))
    if not explicit:
        return hits

    by_file: dict[str, tuple[float, Dict]] = {}
    for hit in hits:
        title = str(hit.get("title") or hit.get("file") or "")
        title_tokens = {_stem_token(t) for t in _token.findall(title.lower())}
        title_tokens = {t for t in title_tokens if t not in DOC_KIND_WORDS and t not in TITLE_GENERIC and t not in STOP and len(t) >= 3}
        if not title_tokens:
            continue
        overlap = len(title_tokens & q_tokens)
        coverage = overlap / len(title_tokens)
        # Tiny tie-break from retrieval score, without letting score overpower title specificity.
        score = coverage + min(0.05, float(hit.get("score") or 0) * 0.01)
        key = str(hit.get("file") or hit.get("source") or title)
        if key not in by_file or score > by_file[key][0]:
            by_file[key] = (score, hit)

    ranked = sorted(by_file.items(), key=lambda kv: kv[1][0], reverse=True)
    if not ranked:
        return hits
    best_key, (best_score, _) = ranked[0]
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
    if best_score < 0.50 or best_score - second_score < 0.15:
        return hits
    focused = [h for h in hits if str(h.get("file") or h.get("source") or h.get("title") or "") == best_key]
    return focused or hits


def _stem_token(token: str) -> str:
    token = (token or "").lower()
    for suffix in ("ibilities", "ibility", "ations", "ation", "ments", "ment", "ing", "ies", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[:-len(suffix)]
    return token


_ANSWER_BOILERPLATE_RE = re.compile(
    r"(?i)^(?:radoncdocs\s*-\s*demo corpus|synthetic documents|synthetic portfolio demo document\.|page\s+\d+\b|document type\b|category\b|owner\b|last updated\b|status\b)"
)


def _clean_answer_text(text: str) -> str:
    """Remove generated document chrome/metadata before returning text to the UI."""
    if not text:
        return ""
    kept: list[str] = []
    for raw_line in str(text).replace("\r", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _ANSWER_BOILERPLATE_RE.match(line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    # Catch metadata-table text that a PDF extractor may flatten onto one line.
    cleaned = re.sub(
        r"(?is)\bDocument type\s+Policy\s*/\s*Procedure\s+Category\s+.*?\bStatus\s+Current(?:\s*-\s*synthetic demo)?\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\bSynthetic portfolio demo document\.[^\n]*", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _query_focused_excerpt(text: str, question: str, max_sentences: int = 3) -> str:
    """Extract the most relevant section/sentences instead of dumping a whole chunk."""
    raw = _clean_answer_text(text)
    if not raw:
        return ""

    q_terms = [_stem_token(t) for t in extract_terms(question)]
    heading_re = re.compile(r"^[A-Z][A-Z0-9 /&()\-]{2,}:?$")
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        body = " ".join(x.strip() for x in buf if x.strip()).strip()
        if body:
            sections.append((heading, body))
        buf = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if heading_re.match(line) and len(line) <= 80:
            flush()
            heading = line.rstrip(":")
        else:
            buf.append(line)
    flush()

    if not sections:
        sections = [("", raw)]

    def stem_hits(value: str) -> int:
        tokens = [_stem_token(t) for t in _token.findall((value or "").lower())]
        return sum(1 for q in set(q_terms) if q and any(t == q or t.startswith(q) or q.startswith(t) for t in tokens))

    scored = []
    for idx, (head, body) in enumerate(sections):
        # Heading matches are especially useful for queries like responsibilities, purpose, or requirements.
        score = 5 * stem_hits(head) + stem_hits(body)
        scored.append((score, idx, head, body))
    score, _, _, block = max(scored, key=lambda x: (x[0], -x[1]))
    if score <= 0:
        block = raw

    # Put inline numbered steps on separate boundaries so they can be selected cleanly.
    block = re.sub(r"\s+(?=\d+[.)]\s+)", "\n", block)
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", block) if x.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences).strip()

    ranked = []
    for idx, sentence in enumerate(sentences):
        ranked.append((stem_hits(sentence), idx, sentence))
    keep = sorted(sorted(ranked, key=lambda x: (x[0], -x[1]), reverse=True)[:max_sentences], key=lambda x: x[1])
    excerpt = " ".join(sentence for _, _, sentence in keep).strip()
    return excerpt or " ".join(sentences[:max_sentences]).strip()



def _source_url(hit: Dict) -> str | None:
    if not SERVE_SOURCE_FILES:
        return None
    raw = str(hit.get("source") or "")
    if not raw:
        return None
    try:
        rel = Path(raw).resolve().relative_to(DATA_DIR.resolve())
        return f"/files/{quote(str(rel).replace('\\', '/'), safe='/')}"
    except Exception:
        return None


def _source_payload(hit: Dict, *, index: int | None = None, snippet: str | None = None) -> dict:
    out = {
        "title": hit.get("title"),
        "file": hit.get("file"),
        "section": hit.get("section") or hit.get("heading"),
        "page": hit.get("page"),
        "page_from": hit.get("page_from"),
        "page_to": hit.get("page_to"),
        "score": float(hit.get("score") or 0),
        "source": hit.get("source"),
        "snippet": _clean_answer_text(snippet if snippet is not None else (hit.get("text") or ""))[:520],
        "category": hit.get("category"),
        "owner": hit.get("owner"),
        "updated": hit.get("updated"),
        "doc_id": hit.get("doc_id"),
        "url": _source_url(hit),
    }
    if index is not None:
        out["index"] = index
    return out

# Optional writer callables (typed for the linter)
compose_answer: Optional[Callable[[str, List[Dict]], str]] = None
sanitize_llm_answer: Optional[Callable[[str, int], Tuple[str, List[int], bool]]] = None

# Import from package re-export to satisfy linters
if ENABLE_WRITER:
    try:
        from rag.writer import compose_answer as _compose, sanitize_llm_answer as _sanitize  # type: ignore
        compose_answer = _compose
        sanitize_llm_answer = _sanitize
    except Exception:
        compose_answer = None
        sanitize_llm_answer = None

@router.post("/answer")
def answer(payload: Ask):
    hits, intent = search_with_context(
        payload.question,
        k=max(12, payload.k or 8),
        min_score=float(payload.min_score or 0.0),
        policy_only=payload.policy_only,
        allowed_doc_types={"policy"} if payload.policy_only else None,
    )

    # Optional folder-level gating. Policy-only mode defaults to the policy folder.
    allow_default = ["policies and procedures"] if payload.policy_only else []
    allow = payload.allowed_paths if payload.allowed_paths is not None else allow_default
    if allow:
        hits = [h for h in hits if _is_allowed(h, allow)]

    terms = extract_terms(payload.question)
    if terms:
        def blob(h): return " ".join([h.get("text") or "", h.get("title") or "", h.get("file") or "", h.get("source") or ""])
        hits = [h for h in hits if term_hits(blob(h), terms) >= 1]

    thr = 0.25 if payload.min_score is None else float(payload.min_score)
    hits = [h for h in hits if float(h.get("score") or 0) >= thr]
    hits = [h for h in hits if _looks_relevant(h, payload.question, min_cov=0.15)]
    hits = _focus_explicit_document(payload.question, hits)

    if not hits:
        return {"found": False, "answer": "I couldn't find a policy-backed answer in the indexed documents.", "sources": [], "mode": "none"}

    allow_proc = q_suggests_procedure(payload.question) \
             or re.match(r"^(how|what should i do|check|set|start|configure|acquire|scan|begin|enable|steps|procedure|troubleshoot)\b", payload.question.strip(), re.I)

    if allow_proc and not re.search(r"\b(who|when|where|how many|how much)\b", payload.question, re.I):
        # Prefer procedure chunks from the single best matching document so
        # ordered steps never get mixed across unrelated SOPs.
        # Retrieval chooses the best document; then inspect *all indexed chunks*
        # from that document so a highly ranked REQUIREMENTS sentence cannot hide
        # the actual PROCEDURE section. This is especially important for
        # troubleshooting questions where the useful answer is an ordered workflow.
        top_file = str(hits[0].get("file") or hits[0].get("source") or "")
        full_doc_hits = all_chunks_for_file(top_file)
        src_hits = full_doc_hits or [h for h in hits if str(h.get("file") or h.get("source") or "") == top_file][:12]
        if not src_hits:
            src_hits = hits[:1]
        procedure_hits = [
            h for h in src_hits
            if re.search(r"\b(PROCEDURE|STEPS|WORKFLOW|CHECKLIST|CHECKS)\b", str(h.get("section") or h.get("heading") or ""), re.I)
        ]
        if procedure_hits:
            src_hits = procedure_hits
        proc_steps = gather_procedure_steps(src_hits, want=8)
        if len(proc_steps) >= 3:
            # Deduplicate source cards by page/section while preserving enough
            # metadata for per-step citations.
            display_hits: list[Dict] = []
            seen_src: set[tuple] = set()
            for h in src_hits:
                key = (h.get("page") or h.get("page_from"), h.get("section") or h.get("heading"))
                if key in seen_src:
                    continue
                seen_src.add(key)
                display_hits.append(h)
            if not display_hits:
                display_hits = src_hits[:1]

            formatted = format_steps_with_cites(proc_steps, display_hits)
            formatted = formatted.split("\n\nSources:\n", 1)[0]
            return {
                "found": True,
                "answer": formatted,
                "sources": [_source_payload(h, index=i + 1) for i, h in enumerate(display_hits)],
                "mode": "procedure-extractive",
            }


    # Call writer only if available; type guard narrows Optional for linter
    if compose_answer is not None and sanitize_llm_answer is not None:
        try:
            raw = compose_answer(payload.question, hits)
            clean, cited_idxs, said_no = sanitize_llm_answer(raw, min(5, len(hits)))
            if said_no or not cited_idxs: raise RuntimeError("writer declined")
            selected = []
            for i in cited_idxs:
                if i-1 < len(hits):
                    h = hits[i-1]
                    selected.append(_source_payload(h, index=i))
            if selected:
                return {"found": True, "answer": clean, "sources": selected, "mode": "rag+writer"}
        except Exception:
            pass

    # Deterministic extractive synthesis for the public demo. Use up to three
    # unique documents so answers look grounded without requiring an LLM.
    selected_hits: list[Dict] = []
    seen_files: set[str] = set()
    for h in hits:
        key = str(h.get("file") or h.get("source") or "")
        if key in seen_files:
            continue
        excerpt = _query_focused_excerpt(h.get("text") or "", payload.question, max_sentences=2)
        if not excerpt:
            continue
        selected_hits.append({**h, "_excerpt": excerpt})
        seen_files.add(key)
        if len(selected_hits) >= 3:
            break

    if not selected_hits:
        selected_hits = [{**hits[0], "_excerpt": _clean_answer_text(hits[0].get("text") or "")}]

    answer_parts = []
    sources = []
    for idx, h in enumerate(selected_hits, start=1):
        excerpt = str(h.get("_excerpt") or "").strip()
        if excerpt:
            answer_parts.append(f"{excerpt} [{idx}]")
        sources.append(_source_payload(h, index=idx, snippet=excerpt[:520]))

    return {
        "found": True,
        "answer": "\n\n".join(answer_parts),
        "sources": sources,
        "mode": "extractive-synthesis" if len(sources) > 1 else "extract-fallback",
    }
