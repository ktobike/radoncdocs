from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch
from api.utils_device import cuda_or_cpu
import re
from typing import List, Dict, Tuple



_CITERE = re.compile(r"\[(\d+)\]")
_TOKEN = re.compile(r"[a-z0-9]+")


def _lex(s: str) -> set[str]:
    return set(_TOKEN.findall((s or "").lower()))


def _shorten(txt: str, max_chars: int = 320) -> str:
    t = (txt or "").strip().replace("\n", " ").replace("  ", " ")
    return t if len(t) <= max_chars else t[:max_chars].rsplit(" ", 1)[0] + "…"


def _strip_artifacts(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\uFEFF", "")
    s = re.sub(r"^##.*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"<\|.*?\|>", "", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


def sanitize_llm_answer(raw: str, max_ref: int) -> tuple[str, List[int], bool]:
    """
    Returns (clean_text, cited_indices, said_not_found)
    - cited_indices are 1-based (matching [1]..[max_ref])
    """
    txt = _strip_artifacts(raw)

    lower = txt.lower()
    said_no = any(
        phrase in lower
        for phrase in [
            "couldn’t find",
            "couldn't find",
            "cannot find",
            "not found",
            "no clear answer in the snippets",
            "not clearly in the snippets",
        ]
    )

    cited: List[int] = []
    for m in _CITERE.finditer(txt):
        n = int(m.group(1))
        if 1 <= n <= max_ref and (n not in cited):
            cited.append(n)

    if not cited:
        said_no = True

    if len(txt) > 1500:
        txt = txt[:1500].rsplit(" ", 1)[0] + "…"

    return txt, cited, said_no


_GEN = None
_TOK = None

def _get_writer(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    global _GEN, _TOK
    if _GEN is not None:
        return _GEN, _TOK
    dev = cuda_or_cpu()
    _TOK = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _GEN = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if dev == "cuda" else None,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    )
    pipe = TextGenerationPipeline(model=_GEN, tokenizer=_TOK, device=0 if dev == "cuda" else -1)
    return pipe, _TOK

def compose_answer(question: str, hits: list[dict]) -> str:
    pipe, tok = _get_writer()
    context = "\n\n".join(f"[{i+1}] {h.get('text','')}" for i, h in enumerate(hits[:6]))
    prompt = (
      "You are a cautious clinical assistant. Answer succinctly and ONLY from the sources.\n"
      "Cite like [1], [2]. If unknown, say you don't know.\n\n"
      f"QUESTION:\n{question}\n\nSOURCES:\n{context}\n\nANSWER:\n"
    )
    out = pipe(prompt, max_new_tokens=300, temperature=0.2, do_sample=False)[0]["generated_text"]
    return out.split("ANSWER:",1)[-1].strip()



