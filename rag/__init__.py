"""RAG utilities package (retrieval, writer, etc.)."""

from .writer import compose_answer, sanitize_llm_answer  # re-export for stable import path

__all__ = [
    "compose_answer",
    "sanitize_llm_answer",
]


