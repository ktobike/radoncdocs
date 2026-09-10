from __future__ import annotations
from typing import List, Dict, Optional
import re, datetime as dt
import os
import html
from pathlib import Path
from typing import Optional
import pymupdf as fitz  # PyMuPDF
from pydantic import BaseModel, Field
from fastapi import APIRouter, Query, HTTPException
import hashlib
from typing import Any
from api.utils_device import cuda_or_cpu
from config import DATA_DIR, SITE_DIR, CACHE_DIR, SERVE_SOURCE_FILES
from demo_metadata import metadata_for_path
from urllib.parse import quote
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
 # for side-effects (Chroma client init)
router = APIRouter(prefix="/journal", tags=["journal"])

# ---- config ----
JOURNAL_ROOT = DATA_DIR / "RedJournal"
SECTION_HEADERS = [
    "ABSTRACT","BACKGROUND","PURPOSE","OBJECTIVE","INTRODUCTION",
    "MATERIALS AND METHODS","PATIENTS AND METHODS","METHODS",
    "RESULTS","DISCUSSION","CONCLUSION","CONCLUSIONS","LIMITATIONS","DATA AVAILABILITY"
]
SEC_RE = re.compile(r"^\s*(" + r"|".join(map(re.escape, SECTION_HEADERS)) + r")\s*$", re.I | re.M)

# -------------------------
# Journal heuristics (regex & cues)
# -------------------------
NUM      = r"(?:\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)"
PCT_RE   = re.compile(rf"{NUM}\s*%")
N_RE     = re.compile(rf"(?:n\s*=\s*|including\s+){NUM}\s*(?:patients|subjects|participants)", re.I)
N_SIMPLE = re.compile(rf"\b({NUM})\s+(?:patients|subjects|participants)\b", re.I)
P_RE     = re.compile(r"\bp\s*[<=>]\s*0\.\d+\b", re.I)
HR_RE    = re.compile(r"\bHR\s*=\s*[\d\.]+(?:\s*\(.*?\))?", re.I)
OR_RE    = re.compile(r"\bOR\s*=\s*[\d\.]+(?:\s*\(.*?\))?", re.I)
RR_RE    = re.compile(r"\bRR\s*=\s*[\d\.]+(?:\s*\(.*?\))?", re.I)
CI_RE    = re.compile(r"\b95%\s*CI\s*\(.*?\)", re.I)
TIME_RE  = re.compile(rf"\b({NUM})\s*(?:mo|month|months|y|year|years|wk|week|weeks|day|days)\b", re.I)
EFFECT_VERBS = ("increased","decreased","reduced","improved","worse","better",
                "associated","predicted","noninferior","superior","inferior",
                "achieved","resulted","significant","correlated","higher","lower")

CLINICAL_CUES = re.compile(r"\b(prospective|retrospective|randomi[sz]ed|cohort|phase\s*\d|hazard ratio|95%\s*CI|AUC|patients)\b", re.I)
NUM_SENTENCE = re.compile(r"(?:\bN\s*=\s*\d+\b|\b\d{2,4}\s+patients\b|HR\s*\(|AUC|95%\s*CI| P\s*=\s*0?\.\d+|%|\bincidence\b)", re.I)

# --- article-type and sentence quality guards -------------------------------
_ARTICLE_NONRESEARCH_TITLE = re.compile(
    r"(?i)\b(cover image|about the cover|issue highlights|editorial|letter to the editor|letters? to the editor|in reply|in regard|gray zone expert opinions?)\b"
)
_ARTICLE_SYSTEMATIC = re.compile(r"(?i)\b(systematic review|meta[- ]analysis|PRISMA|cochrane)\b")
_ARTICLE_RESEARCH_HINT = re.compile(r"(?i)\b(methods?|materials and methods|results?|conclusion[s]?|patients and methods|randomi[sz]ed|prospective|retrospective|cohort|clinical investigation)\b")

_SCISUM = None
_SUM_TOKENIZER = None

BIG_SUMMARY_MODEL = os.getenv("RAD_SUM_MODEL", "allenai/led-large-16384-arxiv")
FALLBACK_SUMMARY_MODEL = "allenai/scitldr"

def _hf_device():
    try:
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

def _get_scisum():
    """
    Prefer LED-16k on GPU; fall back to SciTLDR if LED fails.
    We build model+tokenizer explicitly to avoid the 'input_length is 1' issue.
    """
    global _SCISUM, _SUM_TOKENIZER
    if _SCISUM is not None:
        return _SCISUM

    dev = _hf_device()

    # Try LED first
    try:
        tok = AutoTokenizer.from_pretrained(BIG_SUMMARY_MODEL, use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            BIG_SUMMARY_MODEL,
            torch_dtype=torch.float16 if dev == 0 else None,
        )
        _SCISUM = pipeline("summarization", model=mdl, tokenizer=tok, device=dev)
        _SUM_TOKENIZER = tok
        print(f"[journal] summarizer: {BIG_SUMMARY_MODEL} on {'cuda' if dev==0 else 'cpu'}")
        return _SCISUM
    except Exception as e:
        print(f"[journal] LED load failed: {e}")

    # Fallback to SciTLDR
    try:
        _SCISUM = pipeline("summarization", model=FALLBACK_SUMMARY_MODEL, device=dev)
        _SUM_TOKENIZER = AutoTokenizer.from_pretrained(FALLBACK_SUMMARY_MODEL)
        print(f"[journal] summarizer: {FALLBACK_SUMMARY_MODEL} on {'cuda' if dev==0 else 'cpu'}")
        return _SCISUM
    except Exception as e:
        print(f"[journal] SciTLDR load failed: {e}")
        _SCISUM = None
        _SUM_TOKENIZER = None
        return None

def _chunk_for_summary(text: str, max_tokens: int = 6000, overlap_words: int = 80) -> list[str]:
    """
    Token-aware chunking so LED sees real content (prevents 'input_length is 1').
    """
    if not text:
        return []
    tok = _SUM_TOKENIZER
    if tok is None:
        # best-effort char chunk if tokenizer missing
        return [text[:15000]]

    words = text.split()
    chunks, cur, cur_tok = [], [], 0
    for w in words:
        # encode single word to estimate incremental tokens
        t = tok.encode(" " + w, add_special_tokens=False)
        if cur_tok + len(t) > max_tokens and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_words:] if len(cur) > overlap_words else cur
            # recompute tokens for the carry-over
            cur_tok = len(tok.encode(" ".join(cur), add_special_tokens=False))
        cur.append(w)
        cur_tok += len(t)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def abstractive_boost(long_text: str, max_sentences: int = 2) -> Optional[str]:
    pipe = _get_scisum()
    if not pipe:
        return None
    if not long_text or len(long_text) < 600:
        return None

    pieces = _chunk_for_summary(long_text, max_tokens=6000, overlap_words=80)
    if not pieces:
        return None

    outs = []
    for p in pieces[:2]:  # keep it light: first two chunks
        try:
            o = pipe(
                p,
                max_length=256, min_length=96,
                do_sample=False, truncation=True,
                num_beams=4, no_repeat_ngram_size=3,
            )[0]["summary_text"].strip()
            if o:
                outs.append(o)
        except Exception:
            continue

    if not outs:
        return None

    merged = " ".join(outs)
    # optional short final pass to condense
    try:
        final = pipe(
            merged,
            max_length=200, min_length=80,
            do_sample=False, truncation=True,
            num_beams=4, no_repeat_ngram_size=3,
        )[0]["summary_text"].strip()
    except Exception:
        final = merged

    # sanity: must look like real English
    if len(re.findall(r"[A-Za-z]{3,}", final)) < 10:
        return None

    sents = re.split(r"(?<=[.!?])\s+", final)
    return " ".join(sents[:max_sentences]).strip()
def classify_article(title: str, full_text: str) -> str:
    """
    Returns one of: 'research', 'systematic_review', 'editorial', 'letter', 'cover', 'other'
    """
    t = (title or "").strip()
    body = (full_text or "")
    
    # Title-based first (high precision)
    if re.search(r"(?i)cover", t):
        return "cover"
    if _ARTICLE_NONRESEARCH_TITLE.search(t):
        if re.search(r"(?i)letter", t): return "letter"
        if re.search(r"(?i)editorial", t): return "editorial"
        return "other"

    # Systematic review detection (more specific)
    if _ARTICLE_SYSTEMATIC.search(t):
        return "systematic_review"
    # Only call it systematic if body has strong systematic review language AND it's about this study
    if _ARTICLE_SYSTEMATIC.search(body) and re.search(r"(?i)\b(prisma|cochrane|systematic|meta[- ]analysis)\b", body):
        # Check if it's actually describing this study as a systematic review
        if re.search(r"(?i)\b(this\s+(?:systematic\s+)?review|this\s+meta[- ]analysis|we\s+conducted\s+(?:a\s+)?(?:systematic\s+)?review|we\s+performed\s+(?:a\s+)?meta[- ]analysis)\b", body):
            return "systematic_review"

    # Check for clinical investigation structure first
    has_methods = bool(re.search(r"(?i)\b(methods?|materials and methods|patients and methods)\b", body))
    has_results = bool(re.search(r"(?i)\b(results?)\b", body))
    has_conclusion = bool(re.search(r"(?i)\b(conclusion[s]?|discussion)\b", body))
    
    # If it has clinical structure, it's research
    if has_methods and has_results:
        return "research"
    
    # If body clearly has research hints, call it research
    if _ARTICLE_RESEARCH_HINT.search(body):
        return "research"

    # If no body hints, fall back to title hints
    return "other"

# Filter out junky lines (tables/figures/references/fragments)
_REF_LIKE = re.compile(r"(?i)\b(et al\.|doi:|www\.|https?://|\(\d{4}\)[;.,]|\d{4};\d)")
_TABFIG = re.compile(r"(?i)\b(table\s*\d+|figure\s*\d+|fig\.\s*\d+|supplement|continued)\b")
_VERB_HINT = re.compile(r"(?i)\b(is|are|was|were|showed|observed|enrolled|included|randomi[sz]ed|associated|increased|decreased|improved|reduced|achieved)\b")
def _page_text(page: Any) -> str:
    """Return plain text for a PyMuPDF page; typed as Any to satisfy pyright."""
    try:
        return page.get_text("text")  # modern API
    except Exception:
        # fallbacks across PyMuPDF versions
        if hasattr(page, "getText"):
            return page.getText("text")  # old alias
        try:
            tp = page.get_textpage()
            return tp.extractTEXT()
        except Exception:
            return ""

def is_clean_sentence(s: str) -> bool:
    s1 = (s or "").strip()
    if len(s1) < 40:  # too short → often fragments like "(%) PC1 Amp."
        return False
    if _REF_LIKE.search(s1):
        return False
    if _TABFIG.search(s1[:80]):  # early "Table 1 …", "Figure 2 …"
        return False
    # need at least one verb and end punctuation
    if not _VERB_HINT.search(s1):
        return False
    if not re.search(r"[.!?]\s*$", s1):
        return False
    return True

# def _numeric_priority(s: str) -> int:
#     """Higher is better: AUC/HR/CI highest; % middle; plain P-values lower."""
#     s_low = s.lower()
#     score = 0
#     if "auc" in s_low: score += 5
#     if "hr" in s_low or "hazard ratio" in s_low: score += 5
#     if "95% ci" in s_low: score += 3
#     if "%" in s_low: score += 2
#     if re.search(r"\bp\s*=\s*0?\.\d+", s_low): score += 1
#     return score
ENDPOINT_HINTS = re.compile(r"\b(overall survival|OS|progression[- ]free|PFS|local control|LC|disease[- ]free|DFS|toxicit|grade\s*[3-5])\b", re.I)

def _best_numeric_sentences(results: str, concl: str, max_keep: int = 3) -> list[str]:
    raw = [s for s in (_split_sentences(results) + _split_sentences(concl)) if NUM_SENTENCE.search(s)]
    def pri(s: str) -> int:
        sL = s.lower(); score = 0
        if ENDPOINT_HINTS.search(sL): score += 6      # <--- new: endpoints first
        if "noninferior" in sL or "non-inferior" in sL: score += 3
        if "superior" in sL or "inferior" in sL: score += 2
        if "auc" in sL or "c-index" in sL: score += 5
        if "hr" in sL or "hazard ratio" in sL or "shr" in sL: score += 5
        if "95% ci" in sL: score += 3
        if "%" in sL: score += 2
        if re.search(r"\bp\s*=\s*0?\.\d+", sL): score += 1
        return score

    # clean + rank
    seen, keep = set(), []
    for s in sorted(raw, key=lambda x: (pri(x), -len(x)), reverse=True):
        if not is_clean_sentence(s): 
            continue
        sig = s.strip().lower()
        if sig not in seen:
            seen.add(sig); keep.append(s)
        if len(keep) >= max_keep: break
    return keep

_LIMIT_HEAD = re.compile(r"(?im)^\s*limitations?\s*[:.]?$")
_PAGE_ARTIFACT = re.compile(r"(?im)^\s*(?:page|pg\.?)\s+\d+\s*$")
_SECTION_PREFIX = re.compile(r"(?i)^\s*(?:limitations?|methods?|results?|discussion|conclusions?)\s*[:.-]?\s*")

def _clean_summary_field(value: str | None, *, strip_section_prefix: bool = False) -> str | None:
    """Remove PDF page/header artifacts that should never leak into a summary field."""
    if not value:
        return None
    cleaned = _PAGE_ARTIFACT.sub(" ", value)
    cleaned = re.sub(r"(?i)^\s*(?:page|pg\.?)\s+\d+\s+", "", cleaned)
    if strip_section_prefix:
        cleaned = _SECTION_PREFIX.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None

def _extract_limitations(sections: Dict[str,str]) -> str | None:
    # 1) Explicit section
    for name in ("LIMITATIONS",):
        block = _clean_summary_field(sections.get(name) or "", strip_section_prefix=True) or ""
        if block:
            parts = re.split(r"(?<=[.!?])\s+", block)
            return _clean_summary_field(parts[0] if parts else block, strip_section_prefix=True)
    # 2) First sentence in Conclusions/Discussion mentioning limitation
    for name in ("CONCLUSION","CONCLUSIONS","DISCUSSION"):
        blk = sections.get(name) or ""
        for s in re.split(r"(?<=[.!?])\s+", blk):
            if re.search(r"\blimitation|\blimited by|\bsmall sample|\bobservational|\bretrospective", s, re.I):
                return _clean_summary_field(s, strip_section_prefix=True)
    return None

# ---- helpers ----
def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    idxs = [(m.start(), m.end(), m.group(0).strip()) for m in SEC_RE.finditer(text)]
    if not idxs:
        return {"FULLTEXT": text}
    idxs.append((len(text), len(text), "END"))
    for i in range(len(idxs)-1):
        start, end, header = idxs[i]
        nxt, _, _ = idxs[i+1]
        h = re.sub(r"\s+", " ", header.upper())
        sections[h] = text[end:nxt].strip()
    return sections

def _resolve_pdf(path_or_name: str) -> Path | None:
    """Resolve absolute path, repo-relative, or file name under data/raw."""
    if not path_or_name:
        return None

    p = Path(path_or_name)

    # 1) absolute — restrict to RedJournal subtree only
    if p.is_absolute() and p.exists() and p.suffix.lower() == ".pdf":
        try:
            # ensure it's inside JOURNAL_ROOT
            if JOURNAL_ROOT.resolve() in p.resolve().parents:
                return p
        except Exception:
            return None

    # 2) repo-relative (already under data/raw or any subdir in the repo)
    if p.exists() and p.suffix.lower() == ".pdf":
        # only allow if it sits under RedJournal
        try:
            if JOURNAL_ROOT.resolve() in p.resolve().parents:
                return p
        except Exception:
            pass

    # 3) if it ALREADY starts with data/raw/RedJournal, don't prefix again
    if str(p).startswith(str(JOURNAL_ROOT)) and p.suffix.lower() == ".pdf":
        if p.exists():
            return p

    # 4) try under data/raw/RedJournal
    under = (JOURNAL_ROOT / p).resolve()
    if under.exists() and under.suffix.lower() == ".pdf":
        return under

    # 5) fallback: search by filename anywhere under RedJournal
    matches = list(JOURNAL_ROOT.rglob(p.name)) if JOURNAL_ROOT.exists() else []
    return matches[0] if matches else None

_norm_ws = re.compile(r"\s+")
def _norm(s: str) -> str:
    return _norm_ws.sub(" ", s or "").strip().lower()

def _evidence_with_pages(pdf_path: Path, quotes: list[str]) -> list[str]:
    if not quotes:
        return []
    doc = fitz.open(pdf_path)
    out = []
    try:
        page_texts = [_norm(_page_text(doc.load_page(i))) for i in range(len(doc))]
        for q in quotes:
            nq = _norm(q)
            page_hit = None
            probe = nq[:120] if len(nq) > 160 else nq[: min(100, len(nq))]
            for i, pt in enumerate(page_texts):
                if probe and probe in pt:
                    page_hit = i + 1
                    break
            out.append(f"{q} (p.{page_hit})" if page_hit else q)
    finally:
        doc.close()
    return out
def _render_bullets(s: Optional[str]) -> str:
    if not s:
        return ""
    # split on bullet separator we used above
    parts = [p.strip() for p in s.split("•") if p.strip()]
    if len(parts) <= 1:
        return f"<p><strong>Key findings:</strong> {s}</p>"
    return "<div><strong>Key findings:</strong><ul>" + "".join(f"<li>{p}</li>" for p in parts) + "</ul></div>"

# --- tiny on-disk text cache ---
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key_for(pdf_path: Path) -> str:
    st = pdf_path.stat()
    raw = f"{pdf_path.resolve()}::{st.st_size}::{st.st_mtime_ns}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _read_text_cached(pdf_path: Path) -> str:
    key = _cache_key_for(pdf_path)
    cfile = CACHE_DIR / f"{pdf_path.stem}-{key}.txt"
    if cfile.exists():
        return cfile.read_text(encoding="utf-8", errors="ignore")

    doc = fitz.open(pdf_path)
    try:
        # explicit load_page avoids iterator typing weirdness
        texts = [_page_text(doc.load_page(i)) for i in range(len(doc))]
        text = "\n".join(texts)
    finally:
        doc.close()
    cfile.write_text(text, encoding="utf-8")
    return text


def _extract_text_and_meta(pdf_path: Path) -> tuple[str, str, int, str]:
    doc = fitz.open(pdf_path)
    try:
        meta = doc.metadata or {}
        pages = len(doc)
    finally:
        doc.close()

    text = _read_text_cached(pdf_path)
    title = str(meta.get("title") or "").strip()
    if not title or title.lower() in {"anonymous", "untitled", "none", "null", "microsoft word"}:
        title = pdf_path.stem
    rel = str(pdf_path.relative_to(DATA_DIR.parent)) if pdf_path.is_relative_to(DATA_DIR.parent) else pdf_path.name
    return text, title, pages, rel

def _list_pdfs() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    root = JOURNAL_ROOT if JOURNAL_ROOT.exists() else DATA_DIR
    if root.exists():
        it = JOURNAL_ROOT.rglob("*.pdf") if JOURNAL_ROOT.exists() else []
        for p in it:
            meta = metadata_for_path(p)
            try:
                rel_data = str(p.relative_to(DATA_DIR)).replace("\\", "/")
            except Exception:
                rel_data = p.name
            out.append({
                "file": p.name,
                "path": str(p),
                "title": str(meta.get("title") or p.stem),
                "category": str(meta.get("category") or "Research"),
                "owner": str(meta.get("owner") or "Demo research group"),
                "updated": str(meta.get("updated") or ""),
                "year": meta.get("year"),
                "description": str(meta.get("description") or ""),
                "url": f"/files/{quote(rel_data, safe='/')}" if SERVE_SOURCE_FILES else None,
            })
    out.sort(key=lambda x: (x.get("updated") or "", x.get("title") or ""), reverse=True)
    return out


def _match_pdf(file: Optional[str]=None, path: Optional[str]=None, title_substr: Optional[str]=None) -> Optional[Path]:
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = JOURNAL_ROOT / path
        return p if (p.exists() and p.suffix.lower()==".pdf") else None
    items = _list_pdfs()
    if file:
        for it in items:
            if it["file"] == file:
                return Path(it["path"])
    if title_substr:
        needle = re.sub(r"[\W_]+","-", title_substr.lower()).strip("-")
        for it in items:
            if needle in re.sub(r"[\W_]+","-", it["file"].lower()):
                return Path(it["path"])
    return None

# -------------------------
# Journal summarization helpers
# -------------------------
_NUMY = re.compile(r"\b(HR|OR|RR|AUC|c-?index|95%\s*CI|P\s*[<=>]\s*0?\.\d+|\d+(\.\d+)?%|Gy|cc)\b", re.I)

def _first_non_numeric_sentence(*blocks: str) -> Optional[str]:
    for blk in blocks:
        for s in _split_sentences(_light_clean(blk or "")):
            if is_clean_sentence(s) and not _NUMY.search(s):
                return s
    return None

def _strip_heavy_numerics(s: str) -> str:
    # keep direction, drop stat clutter if needed
    return re.sub(r"\s*(?:\(.*?95%\s*CI.*?\)|\(.*?P\s*[<=>].*?\)|\b(HR|OR|RR)\b\s*=\s*[\d\.]+).*", "", s, flags=re.I).strip()

def _abstract_text(sections: Dict[str,str], full_text: str) -> str:
    for key in ("ABSTRACT","BACKGROUND","INTRODUCTION"):
        t = sections.get(key)
        if t and len(t) > 60:
            return t
    # fallback: first ~3000 chars of first 1–1.5 pages worth of text
    return "\n".join(full_text.splitlines()[:120])

def _results_text(sections: Dict[str,str], full_text: str) -> str:
    for key in ("RESULTS","CONCLUSION","CONCLUSIONS","DISCUSSION"):
        t = sections.get(key)
        if t and len(t) > 60:
            return t
    return "\n".join(full_text.splitlines()[120:360])

def _sentences(text: str) -> list[str]:
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text or "")
    return [re.sub(r"\s+"," ", x).strip() for x in s if len(x.strip()) > 20]

def _article_kind(title: str, full_text: str) -> str:
    t = f"{title}\n{full_text[:800]}".lower()
    if "about the cover" in t: return "cover"
    if t.startswith("in reply") or "in reply to" in t: return "reply"
    if t.startswith("in regard") or "in regard to" in t: return "letter"
    if "editorial" in t or "commentary" in t: return "editorial"
    if "systematic review" in t or "meta-analysis" in t: return "meta"
    if "randomized" in t or "randomised" in t or "phase " in t: return "trial"
    return "original"

def _detect_article_type(title: str, full_text: str, sections: Dict[str,str]) -> str:
    """Enhanced article type detection for better TL;DR routing."""
    t = f"{title}\n{full_text[:1200]}".lower()
    
    # Cover pages
    if "about the cover" in t or "cover illustration" in t:
        return "cover"
    
    # Letters and replies
    if any(phrase in t for phrase in ["in reply to", "in regard to", "letter to the editor", "correspondence"]):
        return "letter"
    
    # Editorials and commentaries
    if any(phrase in t for phrase in ["editorial", "commentary", "perspective", "viewpoint"]):
        return "editorial"
    
    # Systematic reviews and meta-analyses
    if any(phrase in t for phrase in ["systematic review", "meta-analysis", "prisma", "cochrane"]):
        return "meta"
    
    # Clinical trials
    if any(phrase in t for phrase in ["randomized", "randomised", "phase ", "clinical trial"]):
        return "trial"
    
    # Check if it has clinical structure
    has_methods = bool(sections.get("METHODS") or sections.get("MATERIALS AND METHODS") or sections.get("PATIENTS AND METHODS"))
    has_results = bool(sections.get("RESULTS"))
    
    
    # Also check for clinical cues in the text
    has_clinical_cues = bool(CLINICAL_CUES.search(full_text[:2000]))
    
    if (has_methods and has_results) or (has_clinical_cues and has_results):
        return "research"
    
    return "other"

def _find_n(text: str) -> str | None:
    for pat in (N_RE, N_SIMPLE):
        m = pat.search(text)
        if m:
            val = m.group(0)
            # normalize "including 214 patients" -> "n=214"
            m2 = re.search(NUM, val)
            if m2: return f"n={m2.group(0)}"
    return None

def _pick_numeric_findings(candidates: list[str], want: int = 3) -> list[str]:
    scored = []
    for s in candidates:
        score = 0
        score += 2 * bool(PCT_RE.search(s))
        score += 2 * (bool(HR_RE.search(s)) or bool(OR_RE.search(s)) or bool(RR_RE.search(s)))
        score += 1 * bool(P_RE.search(s))
        score += 1 * bool(CI_RE.search(s))
        score += 1 * bool(TIME_RE.search(s))
        score += 1 * any(v in s.lower() for v in EFFECT_VERBS)
        scored.append((score, len(s), s))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    out = [s for _,__,s in scored if s]
    # de-dup very similar openings
    uniq, seen = [], set()
    for s in out:
        k = s[:80].lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
        if len(uniq) >= want: break
    return uniq

def _infer_design(abstract: str, title: str) -> str | None:
    t = f"{title}. {abstract}".lower()
    for tag in ("randomized","randomised","phase 1/2","phase 1","phase 2","phase 3",
                "prospective","retrospective","single-institution","multi-institution","multi-center","multicenter"):
        if tag in t: return tag
    return None

def _infer_limitations(full_text: str, kind: str) -> str:
    t = full_text.lower()
    # explicit limitation section?
    m = re.search(r"(?im)^\s*limitations?\s*[:\n]\s*(.+?)(?:\n[A-Z][A-Z \-/&]{2,}|\Z)", full_text)
    if m:
        s = re.sub(r"\s+", " ", m.group(1)).strip()
        return s[:300]
    # heuristic
    if kind in ("editorial","letter","reply","cover"):
        return "Non-original article (no new cohort or outcomes)."
    hints = []
    if "retrospective" in t: hints.append("retrospective design")
    if "single-institution" in t or "single institution" in t: hints.append("single-institution")
    if "small sample" in t or re.search(r"\bn\s*=\s*(\d|[12]\d)\b", t): hints.append("small sample size")
    if "secondary analysis" in t: hints.append("secondary analysis")
    if "selection bias" in t: hints.append("selection bias")
    if hints:
        return "Limitations include " + ", ".join(hints) + "."
    return "Authors did not clearly state limitations; interpret cautiously."


def _cover_tldr(text: str) -> Dict:
    """Very short blurb for Red Journal cover pages."""
    # Grab the two most informative lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Prefer 'About the Art:' and 'About the Artist:'
    about_art = next((l for l in lines if l.lower().startswith("about the art")), None)
    about_artist = next((l for l in lines if l.lower().startswith("about the artist")), None)
    ev = []
    if about_art: ev.append(about_art)
    if about_artist: ev.append(about_artist)
    # Fallback to any line mentioning Gamma Knife / beams / isodose
    if not ev:
        ev = [l for l in lines if re.search(r"(gamma knife|isodose|beams?)", l, re.I)][:2]
    return {
        "tldr": "Cover illustration (not a research article): brief description of the artwork and artist.",
        "methods": None,
        "key_findings": None,
        "limitations": "Not a research article; no study methods or outcomes.",
        "evidence_spans": ev[:3],
    }

def _non_research_tldr(title: str, full_text: str, article_type: str) -> Dict:
    """Handle letters, replies, editorials, and other non-research content."""
    # Extract first meaningful sentence
    lines = [l.strip() for l in full_text.splitlines() if l.strip() and len(l.strip()) > 20]
    first_sent = lines[0] if lines else ""
    
    # Determine appropriate TL;DR based on type
    if article_type == "letter":
        tldr = f"Letter to the editor: {first_sent[:200]}" if first_sent else "Letter to the editor (no primary data)."
    elif article_type == "editorial":
        tldr = f"Editorial: {first_sent[:200]}" if first_sent else "Editorial commentary (no primary data)."
    elif article_type == "meta":
        tldr = f"Systematic review/meta-analysis: {first_sent[:200]}" if first_sent else "Systematic review (no primary data)."
    else:
        tldr = f"Non-research article: {first_sent[:200]}" if first_sent else "Non-research article (no primary data)."
    
    # Extract any numeric content for evidence
    numeric_lines = []
    for line in lines[:10]:  # Check first 10 lines
        if NUM_SENTENCE.search(line):
            numeric_lines.append(line)
    
    return {
        "tldr": tldr,
        "methods": None,
        "key_findings": None,
        "limitations": f"Not a primary research article ({article_type}); no new cohort or outcomes reported.",
        "evidence_spans": numeric_lines[:2],
    }

def _narrative_tldr(sections: Dict[str,str], title: str, full_text: str) -> Dict:
    # Special handling for cover pages
    if re.search(r"(?i)cover", title) or re.search(r"(?i)about the cover", full_text[:500]):
        return _cover_tldr(full_text)
    
    core_blocks = [
        sections.get("ABSTRACT",""),
        sections.get("CONCLUSION","") or sections.get("CONCLUSIONS",""),
        sections.get("DISCUSSION",""),
        sections.get("INTRODUCTION",""),
    ]
    sents = [s for blk in core_blocks for s in _split_sentences(_light_clean(blk)) if is_clean_sentence(s)]
    # pull 1–2 clean, declarative takeaways + any count/scale sentence
    counts = [s for s in sents if re.search(r"\b\d{2,5}\b.*(patients|cases|courses|centers|years)", s, re.I)]
    top = sents[:2]
    ev = (counts[:1] + top)[:3]
    return {
        "tldr": (top[0] if top else title[:120]),
        "methods": None,
        "key_findings": " ".join(top[:2]) if len(top) >= 2 else None,
        "limitations": "Narrative/experience article; generalizability may be limited.",
        "evidence_spans": [e for e in ev if e],
    }

# Meta-analysis helpers
_META_N = re.compile(r"(?i)\b(included|identified|analy(?:zed|sed))\s+(\d{2,5})\s+(studies|trials)\b")
_META_PAT = re.compile(r"(?i)\b(total of|including)\s+(\d{2,6})\s+(patients|participants)\b")
_POOLED  = re.compile(r"(?i)\b(HR|OR|RR)\s*[, ]?\s*([0-9.]+)\s*\(95%\s*CI[, ]\s*([0-9.]+)\s*[-–]\s*([0-9.]+)\)")

def _meta_tldr(text: str, sections: Dict[str,str]) -> Dict:
    sents = [s for s in _split_sentences(text) if is_clean_sentence(s)]
    nstud = next((f"{m.group(2)} {m.group(3)}" for s in sents if (m:=_META_N.search(s))), None)
    npat  = next((f"{m.group(2)} {m.group(3)}" for s in sents if (m:=_META_PAT.search(s))), None)
    pooled = next((f"{m.group(1)} {m.group(2)} (95% CI {m.group(3)}–{m.group(4)})"
                   for s in sents if (m:=_POOLED.search(s))), None)
    concl_blk = sections.get("CONCLUSION") or sections.get("CONCLUSIONS") or ""
    concl_s   = next((s for s in _split_sentences(concl_blk) if is_clean_sentence(s)), "")
    tldr = "Meta-analysis"
    if nstud: tldr += f"; {nstud}"
    if npat:  tldr += f"; {npat}"
    if pooled: tldr += f": pooled {pooled}"
    return {
        "tldr": tldr if len(tldr) > 14 else (concl_s or (sections.get("ABSTRACT","").split(". ")[0] if sections.get("ABSTRACT") else "")),
        "methods": "systematic review/meta-analysis",
        "key_findings": pooled or concl_s or None,
        "limitations": _extract_limitations(sections) or "Meta-analysis; heterogeneity/publication bias may apply.",
        "evidence_spans": [x for x in [nstud, npat, pooled, concl_s] if x],
    }

def _clinical_tldr(sections: Dict[str, str]) -> Dict:
    abstract = sections.get("ABSTRACT") or sections.get("BACKGROUND") or ""
    methods  = sections.get("METHODS") or sections.get("MATERIALS AND METHODS") or ""
    results  = sections.get("RESULTS") or ""
    concl    = sections.get("CONCLUSION") or sections.get("CONCLUSIONS") or ""

    # design & N
    design = None
    if re.search(r"(?i)\b(randomi[sz]ed|phase\s*[0-4]|prospective|retrospective|cohort|case[- ]control|trial)\b", abstract + methods):
        m = re.search(r"(?i)\b(phase\s*[0-4])\b", abstract + methods)
        if m: design = m.group(1).lower()
        if re.search(r"(?i)randomi[sz]ed", abstract + methods):
            design = (design + " randomized" if design else "randomized")

    n_sent = None
    mN = re.search(r"(?i)\b(?:n\s*=\s*|enrolled\s*)(\d{2,5})\b", abstract + methods + results)
    if mN:
        n_sent = f"n={mN.group(1)}"

    method_sentences = [
        cleaned for sentence in _sentence_split(methods)[:2]
        if (cleaned := _clean_summary_field(sentence))
    ]
    methods_summary = " ".join(method_sentences).strip() or design

    numeric = _best_numeric_sentences(results, concl, max_keep=3)

    # NEW: find a non-numeric directional sentence for TL;DR if possible
    nonnum = _first_non_numeric_sentence(concl, abstract, results)
    prefix_bits = []
    if design: prefix_bits.append(design.capitalize())
    if n_sent: prefix_bits.append(n_sent)
    prefix = (": ".join(prefix_bits) + ": ") if prefix_bits else ""

    if nonnum:
        tldr = (prefix + _strip_heavy_numerics(nonnum))
    elif numeric:
        # last resort: use numeric, but strip the heaviest bits
        tldr = (prefix + _strip_heavy_numerics(numeric[0]))
    else:
        tldr = (prefix + (abstract.split(". ")[0].strip() if abstract else ""))

    # Key findings = numbers only (2–3 bullets joined with dots)
    key_list = numeric[:3]
    key = " • ".join(key_list) if key_list else None

    limits = _extract_limitations(sections) or \
             "Observational journal article; limitations may include cohort size and generalizability unless otherwise stated."

    ev = []
    if design: ev.append(design)
    if n_sent: ev.append(n_sent)
    ev.extend(numeric[:2])

    return {
        "tldr": tldr.strip(),
        "methods": methods_summary,
        "key_findings": key,
        "limitations": limits,
        "evidence_spans": [s for s in ev if s],
    }

def _tldr_for_pdf(pdf: Path, mode: str = "extractive", sentences: int = 5):
    # 1) Read & sectionize
    text, title, pages, rel_src = _extract_text_and_meta(pdf)
    sections = _split_sections(text)
    is_policy = "policies and procedures" in (rel_src or "").lower()

    # 2) Build the base summary dict FIRST (so `s` always exists)
    # classify_article is already defined above in your file
    if is_policy:
        s: Dict[str, Any] = _policy_tldr(text)
    else:
        kind = classify_article(title, text)
        if kind in ("editorial", "letter", "cover", "other"):
            s = _narrative_tldr(sections, title, text)
        elif kind == "systematic_review":
            s = _meta_tldr(text, sections)
        else:
            s = _clinical_tldr(sections)

    # 3) Abstractive override (only for the TL;DR line)
    long_core = " ".join(filter(None, [
        sections.get("RESULTS",""),
        sections.get("CONCLUSIONS") or sections.get("CONCLUSION",""),
        sections.get("ABSTRACT","")
    ])).strip()

    # Only attempt abstractive if there is real content
    def _looks_nontrivial(txt: str) -> bool:
        # at least some alphabetic words and reasonable length
        return bool(re.search(r"[A-Za-z]{3}", txt)) and len(txt) >= 600

    if mode == "abstractive" and _looks_nontrivial(long_core):
        abs_text = abstractive_boost(long_core)  # may be None
        def _looks_english(s: str) -> bool:
            if not s: return False
            # reject if mostly non-letters or too short
            if len(s) < 60: return False
            # must contain vowels and spaces/punctuation
            return bool(re.search(r"[AEIOUaeiou].*[AEIOUaeiou]", s)) and bool(re.search(r"[ .,:;!?-]", s))
        if abs_text and _looks_english(abs_text):
            sents = re.split(r"(?<=[.!?])\s+", abs_text)
            if sents:
                s["tldr"] = " ".join(sents[:2]).strip()

    # 4) Final presentation cleanup. PDF page banners and repeated section
    # headings are extraction artifacts, not summary content.
    for field in ("tldr", "methods", "key_findings", "limitations"):
        cleaned = _clean_summary_field(s.get(field), strip_section_prefix=(field == "limitations"))
        if cleaned is not None:
            s[field] = cleaned

    # Attach page-tagged evidence safely.
    clean_evidence = []
    for evidence in s.get("evidence_spans", []):
        cleaned = _clean_summary_field(str(evidence))
        if cleaned:
            clean_evidence.append(cleaned)
    s["evidence_spans"] = _evidence_with_pages(pdf, clean_evidence)

    # 5) Return the full payload
    return {
        "title": title,
        "pages": pages,
        "rel_src": rel_src,
        "summary": s,
    }



   

def _split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [re.sub(r"\s+"," ", s).strip() for s in sents if s and len(s)>40]

def _extractive(text: str, sentences: int = 5) -> List[str]:
    sents = _split_sentences(text)
    if not sents: return []
    KEYWORDS = ("randomized","prospective","methods","results","conclusion",
                "objective","background","materials and methods","trial",
                "toxicity","dose","LET","proton","SRS","escalation")
    def score(s: str) -> float:
        kw = sum(1 for k in KEYWORDS if re.search(k, s, re.I))
        ln = len(s); center = 180
        return kw + (1.0 - min(1.0, abs(ln-center)/center))
    ranked = sorted(sents, key=score, reverse=True)
    out, seen = [], set()
    for s in ranked:
        sig = s.lower()
        if sig not in seen:
            seen.add(sig); out.append(s)
        if len(out) >= sentences: break
    return out or sents[:sentences]

_SUMMARIZER = None
_TOKENIZER = None
def _get_abstractive_pipeline(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    from transformers import pipeline
    import torch
    dev = 0 if torch.cuda.is_available() else -1
    _SUMMARIZER = pipeline("summarization", model=model_name, tokenizer=_TOKENIZER, device=dev, framework="pt")
    return _SUMMARIZER, _TOKENIZER


def _chunk_text_by_tokens(text: str, tokenizer, max_tokens: int = 900) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur, cur_tok = [], [], 0
    for p in paras:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if cur_tok + len(ids) > max_tokens and cur:
            chunks.append(" ".join(cur)); cur, cur_tok = [], 0
        cur.append(p); cur_tok += len(ids)
    if cur: chunks.append(" ".join(cur))
    out = []
    for c in chunks:
        if len(tokenizer.encode(c, add_special_tokens=False)) <= max_tokens:
            out.append(c)
        else:
            step = max(1500, len(c)//2)
            out.extend([c[i:i+step] for i in range(0, len(c), step)])
    return out

def _abstractive(text: str, sentences: int = 5, model_name: str = "sshleifer/distilbart-cnn-12-6") -> List[str]:
    summarizer, tok = _get_abstractive_pipeline(model_name)
    chunks = _chunk_text_by_tokens(text, tok, max_tokens=900)
    parts = []
    max_len = min(220, 30 * max(1, sentences))
    min_len = min(160, 22 * max(1, sentences))
    for c in chunks[:6]:
        parts.append(summarizer(c, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)[0]["summary_text"].strip())
    merged = " ".join(parts) if parts else text
    final = summarizer(merged, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)[0]["summary_text"].strip()
    sents = _split_sentences(final)
    return sents[:sentences] if sents else [final]

def _extractive_tldr(sections: Dict[str,str]) -> Dict:
    abstract = sections.get("ABSTRACT") or sections.get("BACKGROUND") or ""
    conclusions = sections.get("CONCLUSION") or sections.get("CONCLUSIONS") or ""
    results = sections.get("RESULTS") or ""
    methods = sections.get("METHODS") or sections.get("MATERIALS AND METHODS") or ""
    def first_s(block: str, n=2):
        s = _sentence_split(block); return " ".join(s[:n]) if s else ""
    evidence = []
    for block in [abstract, methods, results, conclusions]:
        for s in _sentence_split(block)[:2]:
            if len(evidence) < 6: evidence.append(s)
    return {
        "tldr": first_s(abstract, 2) or first_s(conclusions, 2) or first_s(results, 2),
        "methods": first_s(methods, 2),
        "key_findings": first_s(results, 3),
        "limitations": "Limitations not explicitly stated in extract; interpret cautiously.",
        "evidence_spans": evidence,
    }
# --- SOP/POLICY fallback helpers ---------------------------------------------

SOP_HEADINGS = (
    "PURPOSE", "OBJECTIVE", "SCOPE", "RESPONSIBILITIES", "ROLES",
    "EQUIPMENT", "MATERIALS", "PROCEDURE", "STEPS", "REFERENCES", "DEFINITIONS"
)

_bullet_re = re.compile(r"^\s*((\d+[\.\)]|[A-Za-z][\.\)]|[-•▪‣*])\s+)(.+)$")
_heading_val_re = re.compile(r"(?im)^\s*({heads})\s*[:\-]?\s*(.+)$".format(
    heads="|".join(map(re.escape, SOP_HEADINGS))
))
# --- light text cleanup (fix hyphen line breaks, spaces, newlines) ---
_hyphen_break = re.compile(r"(\w)-\n(\w)")
_newline_run  = re.compile(r"[ \t]*\n[ \t]*")
_space_run    = re.compile(r"[ \t]{2,}")

def _light_clean(s: str) -> str:
    if not s: return ""
    s = _hyphen_break.sub(r"\1\2", s)
    s = _newline_run.sub("\n", s)
    s = _space_run.sub(" ", s)
    return s.strip()


# --- find a section's text between this heading and the next heading ---
def _section_block(full_text: str, name: str) -> str | None:
    # case-insensitive exact heading on its own line
    pat = re.compile(rf"(?im)^\s*{re.escape(name)}\s*$")
    heads = list(pat.finditer(full_text))
    if not heads:
        return None
    start = heads[0].end()
    # find next ALL-CAPS style heading (common in SOPs)
    next_head = re.search(r"(?m)^\s*[A-Z][A-Z \-/&]{2,}\s*$", full_text[start:])
    end = start + next_head.start() if next_head else len(full_text)
    return full_text[start:end].strip()

def _grab_heading_value(text: str, *names: str) -> Optional[str]:
    """Return the first 'HEADING: value' line that matches any of `names`."""
    for name in names:
        m = re.search(rf"(?im)^\s*{re.escape(name)}\s*[:\-]?\s*(.+)$", text)
        if m:
            return m.group(1).strip()
    return None

_META_LINE = re.compile(r"""(?im)^\s*
    (title|policy\s*number|doc(ument)?\s*id|effective\s*date|revision|review(ed|er)?\s*by|
     approved\s*by|contributors?|contributor|department|site)\s*[:\-]
""", re.X)

def _looks_meta(line: str) -> bool:
    l = (line or "").strip()
    return bool(_META_LINE.match(l)) or (l.isupper() and ":" in l and len(l) < 80)

# optional: simple imperative cue (helps pick true steps)
_IMPERATIVE = re.compile(r"^(place|click|enter|verify|select|start|create|determine|position|scan|record|confirm|attach|open|choose|press)\b", re.I)

def _policy_tldr(full_text: str) -> dict:
    txt = _light_clean(full_text)

    tldr = _grab_heading_value(txt, "PURPOSE", "OBJECTIVE")
    if not tldr:
        first_lines = [l for l in txt.splitlines() if l.strip() and not _looks_meta(l)]
        tldr = first_lines[0][:280] if first_lines else ""

    methods = _grab_heading_value(txt, "RESPONSIBILITIES","ROLES","SCOPE","EQUIPMENT","MATERIALS") or None

    proc = _section_block(txt, "PROCEDURE") or _section_block(txt, "STEPS") or txt

    steps: list[str] = []
    for line in proc.splitlines():
        if _looks_meta(line): 
            continue
        m = _bullet_re.match(line)
        if m:
            cand = m.group(3).strip()
            if not _looks_meta(cand):
                steps.append(cand)
                if len(steps) >= 8: break

    if not steps:
        # fallback: pick imperative-looking lines, else first informative lines
        cands = [l.strip() for l in proc.splitlines() 
                 if len(l.strip()) > 20 and not _looks_meta(l)]
        imps = [l for l in cands if _IMPERATIVE.search(l)]
        steps = (imps or cands)[:6]

    key_findings = " ".join([s[:200].rstrip(",; ") for s in steps[:4]]) if steps else None
    evidence = steps[:4]

    return {
        "tldr": tldr or "",
        "methods": methods,
        "key_findings": key_findings,
        "limitations": "Policy/SOP: limitations typically not listed; follow documented steps and local safety checks.",
        "evidence_spans": evidence,
    }


# ---- models ----
class TLDRResponse(BaseModel):
    title: str
    pages: int
    tldr: str
    methods: str | None = None
    key_findings: str | None = None
    limitations: str | None = None
    evidence_spans: list[str] = Field(default_factory=list)
    source_file: str

# ---- routes ----
@router.get("/list")
def journal_list():
    return _list_pdfs()

@router.get("/tldr")
def journal_tldr(
    file: Optional[str] = Query(None, description="Exact file name from /journal/list"),
    path: Optional[str] = Query(None, description="Full path from /journal/list"),
    sentences: int = Query(5, ge=1, le=12),
    mode: str = Query("extractive")
):
    target = path or file
    pdf = _resolve_pdf(target or "")
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found under the configured demo data directory")

    out = _tldr_for_pdf(pdf, mode=mode, sentences=sentences)
    s = out["summary"]

    return TLDRResponse(
        title=out["title"],
        pages=out["pages"],
        tldr=s.get("tldr", ""),
        methods=s.get("methods"),
        key_findings=s.get("key_findings"),
        limitations=s.get("limitations"),
        evidence_spans=s.get("evidence_spans", []),
        source_file=out["rel_src"],
    )



@router.get("/tldr_share")
def journal_tldr_share(
    file: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    mode: str = Query("extractive", pattern="^(extractive|abstractive)$"),
    sentences: int = Query(5, ge=1, le=12),
):
    target = path or file
    pdf = _resolve_pdf(target or "")
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found under the configured demo data directory")

    out = _tldr_for_pdf(pdf, mode=mode, sentences=sentences)
    s = out["summary"]

    safe_methods = html.escape(str(s.get("methods") or ""))
    safe_findings = html.escape(str(s.get("key_findings") or ""))
    safe_limitations = html.escape(str(s.get("limitations") or ""))
    safe_evidence = [html.escape(str(e)) for e in s.get("evidence_spans", [])]

    evidence_html = "".join(f'<blockquote>{e}</blockquote>' for e in safe_evidence)
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Summary - {html.escape(out['title'])}</title>
<style>
:root{{--text:#172033;--muted:#667085;--line:#e4e7ec;--brand:#2563eb;--bg:#f5f7fa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.top{{height:58px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 24px;font-size:13px;font-weight:700}}
main{{max-width:860px;margin:42px auto 72px;padding:0 22px}}
h1{{font-size:28px;line-height:1.25;letter-spacing:-.025em;margin:0 0 8px}}
.meta{{color:var(--muted);font-size:12px;margin-bottom:30px}}
.section{{display:grid;grid-template-columns:135px 1fr;gap:24px;padding:21px 0;border-top:1px solid var(--line)}}
.label{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.055em;color:#475467}}
.value{{font-size:14px;line-height:1.72;color:#344054}}
blockquote{{margin:0 0 10px;border-left:2px solid #b2ccff;padding:6px 0 6px 12px;color:#475467;font-size:12px;line-height:1.6}}
.footer{{border-top:1px solid var(--line);padding-top:18px;margin-top:8px;color:#98a2b3;font-size:11px}}
@media(max-width:640px){{.section{{grid-template-columns:1fr;gap:7px}}main{{margin-top:28px}}}}
</style></head><body>
<div class="top">RadOncDocs · Research summary</div>
<main>
  <h1>{html.escape(out['title'])}</h1>
  <div class="meta">{out['pages']} pages · {html.escape(out['rel_src'])}</div>
  <div class="section"><div class="label">TL;DR</div><div class="value">{html.escape(s.get('tldr',''))}</div></div>
  {f'<div class="section"><div class="label">Methods</div><div class="value">{safe_methods}</div></div>' if safe_methods else ''}
  {f'<div class="section"><div class="label">Key findings</div><div class="value">{safe_findings}</div></div>' if safe_findings else ''}
  {f'<div class="section"><div class="label">Limitations</div><div class="value">{safe_limitations}</div></div>' if safe_limitations else ''}
  {f'<div class="section"><div class="label">Evidence</div><div class="value">{evidence_html}</div></div>' if evidence_html else ''}
  <div class="footer">Generated by the RadOncDocs portfolio demo · {dt.datetime.now().isoformat(timespec='seconds')}</div>
</main></body></html>"""

    out_dir = SITE_DIR / "summaries"; out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+","-", pdf.stem.lower()).strip("-")
    fname = f"{slug}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    (out_dir / fname).write_text(html_doc, encoding="utf-8")
    return {"url": f"/site/summaries/{fname}"}

@router.get("/digest")
def journal_digest(
    max_papers: int = Query(200, ge=1, le=1000),
    mode: str = Query("extractive", pattern="^(extractive|abstractive)$"),
    sentences: int = Query(5, ge=1, le=12),
):

    items = _list_pdfs()[:max_papers]
    cards: List[str] = []

    for it in items:
        pdf = Path(it["path"])
        try:
            out = _tldr_for_pdf(pdf, mode=mode, sentences=sentences)
            s = out["summary"]
            if not s.get("tldr"):  # skip empty/garbage cards
                continue
            safe_title = html.escape(str(out.get("title") or ""))
            safe_tldr = html.escape(str(s.get("tldr") or ""))
            safe_methods = html.escape(str(s.get("methods") or ""))
            safe_findings = html.escape(str(s.get("key_findings") or ""))
            safe_limitations = html.escape(str(s.get("limitations") or ""))
            cards.append(f"""
<div class="paper">
  <div class="title">{safe_title}</div>
  <div class="meta">{out['pages']} pages · Source: {html.escape(out['rel_src'])}</div>
  <div class="sec"><b>TL;DR:</b> {safe_tldr}</div>
  {f"<div class='sec'><b>Methods:</b> {safe_methods}</div>" if safe_methods else ""}
  {f"<div class='sec'><b>Key findings:</b> {safe_findings}</div>" if safe_findings else ""}
  <div class="sec"><b>Limitations:</b> {safe_limitations}</div>
</div>""")
        except Exception:
            continue

    out_dir = SITE_DIR / "summaries"; out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"digest-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RadOncDocs Digest</title>
<style>
:root{{--text:#172033;--muted:#667085;--line:#e4e7ec;--bg:#f5f7fa}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.top{{height:58px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 24px;font-size:13px;font-weight:700}}
main{{max-width:960px;margin:40px auto 72px;padding:0 22px}}h1{{font-size:28px;letter-spacing:-.025em;margin:0}}
.deck-meta{{color:var(--muted);font-size:12px;margin:8px 0 26px}}
.paper{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:20px 22px;margin:12px 0}}
.title{{font-weight:700;font-size:15px}}.meta{{color:var(--muted);font-size:11px;margin:5px 0 15px}}
.sec{{font-size:13px;line-height:1.65;color:#475467;padding:7px 0}}.sec b{{color:#344054}}
</style></head><body><div class="top">RadOncDocs · Research digest</div><main>
<h1>Research digest</h1><div class="deck-meta">{len(cards)} paper{'s' if len(cards) != 1 else ''} · Generated {dt.date.today().isoformat()}</div>
{"".join(cards) if cards else "<p>No papers were available to summarize.</p>"}
</main></body></html>"""
    (out_dir / fname).write_text(html_doc, encoding="utf-8")
    return {"url": f"/site/summaries/{fname}", "count": len(cards)}
