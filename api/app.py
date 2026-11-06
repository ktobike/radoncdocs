from __future__ import annotations

from pathlib import Path
from typing import List
from urllib.parse import quote
import pymupdf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma

from api.qa_routes import router as qa_router
from api.journal_routes import router as journal_router
from config import (
    DATA_DIR,
    UI_DIR,
    SITE_DIR,
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION,
    ENABLE_ADMIN,
    SERVE_SOURCE_FILES,
)
from ingest.ingest import build_index, index_pdf, delete_pdf_from_index, get_embeddings
from demo_metadata import metadata_for_path

app = FastAPI(
    title="RadOncDocs API",
    description="Source-cited document retrieval and research summarization workspace.",
    version="0.8.1",
)

# Same-origin UI. Avoid permissive CORS in the public demo.
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
app.mount("/site", StaticFiles(directory=str(SITE_DIR), html=True), name="site")
if SERVE_SOURCE_FILES:
    app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


def _require_admin() -> None:
    if not ENABLE_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Document administration is disabled. Set RADONCDOCS_ENABLE_ADMIN=1 for local development.",
        )


def _safe_under_raw(p: Path) -> Path:
    """Resolve a path and ensure it stays under the configured data directory."""
    base = DATA_DIR.resolve()
    rp = (base / p).resolve() if not p.is_absolute() else p.resolve()
    if base not in rp.parents and rp != base:
        raise HTTPException(status_code=400, detail="Path must be under the data directory")
    return rp


def _list_pdfs_under(root: Path) -> list[dict]:
    out = []
    for p in root.rglob("*.pdf"):
        st = p.stat()
        rel = str(p.relative_to(DATA_DIR)).replace("\\", "/")
        meta = metadata_for_path(p)
        try:
            with pymupdf.open(p) as pdf:
                pages = len(pdf)
                pdf_title = str((pdf.metadata or {}).get("title") or "").strip()
        except Exception:
            pages = None
            pdf_title = ""
        out.append(
            {
                "id": meta.get("id") or p.stem,
                "file": p.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "rel": rel,
                "title": meta.get("title") or pdf_title or p.stem,
                "doc_type": meta.get("doc_type") or ("journal" if "redjournal" in rel.lower() else "policy"),
                "category": meta.get("category") or "Uncategorized",
                "owner": meta.get("owner") or "Demo workspace",
                "updated": meta.get("updated") or "",
                "status": meta.get("status") or "Current",
                "description": meta.get("description") or "",
                "tags": meta.get("tags") or [],
                "year": meta.get("year"),
                "pages": pages,
                "url": f"/files/{quote(rel, safe='/')}" if SERVE_SOURCE_FILES else None,
            }
        )
    out.sort(key=lambda x: (x.get("updated") or "", x.get("title") or ""), reverse=True)
    return out


@app.get("/docs/list")
def docs_list(subdir: str | None = Query(None, description="Optional subfolder under the data directory")):
    root = _safe_under_raw(Path(subdir)) if subdir else DATA_DIR
    return _list_pdfs_under(root) if root.exists() else []


@app.get("/workspace/stats")
def workspace_stats():
    docs = _list_pdfs_under(DATA_DIR)
    policies = [d for d in docs if d.get("doc_type") == "policy"]
    journals = [d for d in docs if d.get("doc_type") == "journal"]
    categories = sorted({str(d.get("category") or "") for d in docs if d.get("category")})
    return {
        "documents": len(docs),
        "policies": len(policies),
        "journals": len(journals),
        "categories": len(categories),
        "latest_updated": max((d.get("updated") or "" for d in docs), default=""),
    }


@app.post("/docs/upload")
async def docs_upload(files: List[UploadFile] = File(...), dest: str = Form("uploads")):
    _require_admin()
    target_dir = _safe_under_raw(Path(dest))
    target_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        # Strip any path components supplied by the client.
        safe_name = Path(upload.filename).name
        if not safe_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {safe_name}")

        target = target_dir / safe_name
        base, ext, i = target.stem, target.suffix, 1
        while target.exists():
            target = target_dir / f"{base} ({i}){ext}"
            i += 1

        target.write_bytes(await upload.read())
        try:
            chunks, doc_type = index_pdf(target)
            results.append(
                {"rel": target.relative_to(DATA_DIR).as_posix(), "chunks_indexed": chunks, "doc_type": doc_type}
            )
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Indexing failed for {safe_name}: {exc}") from exc

    return {"ok": True, "count": len(results), "results": results}


def _vs() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


@app.delete("/docs")
def docs_delete(rel_path: str = Query(..., description="Path relative to the data directory")):
    _require_admin()
    abs_path = _safe_under_raw(Path(rel_path))
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        _vs().delete(where={"source": str(abs_path.resolve())})
    except Exception:
        # The JSONL cleanup below still removes the document from the lexical index.
        pass

    try:
        delete_pdf_from_index(abs_path.name)
    finally:
        abs_path.unlink(missing_ok=True)

    return {"ok": True, "deleted": abs_path.relative_to(DATA_DIR).as_posix()}


@app.post("/index/rebuild")
def index_rebuild():
    _require_admin()
    try:
        build_index(rebuild=True)
        return {"ok": True, "rebuild": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(qa_router)
app.include_router(journal_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {
        "ok": True,
        "admin_enabled": ENABLE_ADMIN,
        "index_ready": CHUNKS_PATH.exists() and CHROMA_DIR.exists(),
        "document_count": len(_list_pdfs_under(DATA_DIR)),
        "version": "0.8.1",
    }
