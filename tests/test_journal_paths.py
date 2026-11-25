"""The journal routes must only ever open PDFs that live inside the journal folder."""
from __future__ import annotations

import os

import pytest

import api.journal_routes as jr


@pytest.fixture()
def journal_root(tmp_path, monkeypatch):
    """An isolated journal folder with one paper, plus a PDF outside it."""
    root = tmp_path / "data" / "RedJournal"
    (root / "2024").mkdir(parents=True)
    (root / "2024" / "Inside Paper.pdf").write_bytes(b"%PDF-1.4 inside")
    (root / "notes.txt").write_text("not a pdf")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"%PDF-1.4 outside")
    monkeypatch.setattr(jr, "JOURNAL_ROOT", root)
    return root, outside / "secret.pdf"


def test_bare_file_name_resolves_anywhere_under_journal_root(journal_root):
    root, _ = journal_root
    assert jr._resolve_pdf("Inside Paper.pdf") == (root / "2024" / "Inside Paper.pdf").resolve()


def test_relative_and_absolute_paths_inside_root_resolve(journal_root):
    root, _ = journal_root
    expected = (root / "2024" / "Inside Paper.pdf").resolve()
    assert jr._resolve_pdf("2024/Inside Paper.pdf") == expected
    assert jr._resolve_pdf(str(expected)) == expected


@pytest.mark.parametrize(
    "make_request",
    [
        pytest.param(lambda root, secret: os.path.relpath(secret, root), id="dot-dot traversal"),
        pytest.param(lambda root, secret: str(secret), id="absolute path outside"),
        pytest.param(lambda root, secret: f"2024/../../../private/{secret.name}", id="traversal via subdir"),
    ],
)
def test_paths_outside_journal_root_are_refused(journal_root, make_request):
    root, secret = journal_root
    assert jr._resolve_pdf(make_request(root, secret)) is None


def test_symlink_pointing_outside_is_refused(journal_root):
    root, secret = journal_root
    link = root / "link.pdf"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not supported on this filesystem")
    assert jr._resolve_pdf("link.pdf") is None


@pytest.mark.parametrize("name", ["notes.txt", "*.pdf", "**/*.pdf", "missing.pdf", "", "bad\x00name.pdf"])
def test_non_pdfs_globs_and_garbage_are_refused(journal_root, name):
    assert jr._resolve_pdf(name) is None
