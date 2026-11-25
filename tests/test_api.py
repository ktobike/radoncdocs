"""HTTP-level checks against the app with the bundled synthetic corpus."""
from __future__ import annotations

import json
import os

import pymupdf
import pytest
from fastapi.testclient import TestClient

import api.journal_routes as jr
from api.app import app
from config import CACHE_DIR, CHROMA_DIR, CHUNKS_PATH, DATA_DIR

client = TestClient(app)


def _assert_no_server_paths(payload) -> None:
    blob = json.dumps(payload)
    assert str(DATA_DIR) not in blob, "response leaks an absolute server path"
    assert '"path":' not in blob and '"source":' not in blob


@pytest.mark.parametrize("route", ["/docs/list", "/journal/list", "/workspace/stats", "/health"])
def test_listing_routes_do_not_expose_server_paths(route):
    response = client.get(route)
    assert response.status_code == 200
    _assert_no_server_paths(response.json())


def test_journal_list_entries_still_usable_by_the_ui():
    papers = client.get("/journal/list").json()
    assert papers, "synthetic corpus should include research papers"
    for paper in papers:
        assert paper["file"].endswith(".pdf")
        assert paper["url"].startswith("/files/RedJournal/")


@pytest.mark.parametrize("route", ["/journal/tldr", "/journal/tldr_share"])
def test_journal_routes_refuse_pdfs_outside_the_journal_folder(route, tmp_path):
    # A real, parseable PDF, so a regression would actually extract and cache its text.
    outside = tmp_path / "outside-canary.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "RESULTS:\nCANARY text that must never leave the server.")
        doc.save(outside)
    for value in (os.path.relpath(outside, jr.JOURNAL_ROOT), str(outside)):
        for param in ("file", "path"):
            response = client.get(route, params={param: value})
            assert response.status_code == 404, (param, value)
    assert not list(CACHE_DIR.glob("outside-canary-*.txt")), "outside PDF text must never be extracted"


def test_journal_summary_for_a_listed_paper():
    paper = client.get("/journal/list").json()[0]
    response = client.get("/journal/tldr", params={"file": paper["file"], "mode": "extractive"})
    assert response.status_code == 200
    body = response.json()
    assert body["tldr"]
    assert not os.path.isabs(body["source_file"])


index_required = pytest.mark.skipif(
    not (CHUNKS_PATH.exists() and CHROMA_DIR.exists()),
    reason="search index not built; run `python -m ingest.ingest --rebuild` (CI builds it first)",
)


@index_required
def test_answer_sources_use_relative_paths():
    response = client.post(
        "/answer",
        json={"question": "What checks are required before a dataset is frozen for evaluation?", "policy_only": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    _assert_no_server_paths(body)
    for source in body["sources"]:
        assert source["rel"].startswith("Policies and Procedures/")
        assert source["url"].startswith("/files/Policies%20and%20Procedures/")


