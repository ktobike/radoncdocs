"""End-to-end smoke test for a locally running RadOncDocs demo.

Run after indexing and starting the app:
    python scripts/smoke_test.py

Use a non-default port if needed:
    $env:RADONCDOCS_BASE_URL="http://127.0.0.1:8010"   # PowerShell
    RADONCDOCS_BASE_URL=http://127.0.0.1:8010 python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE = os.getenv("RADONCDOCS_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def get_json(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {label}{': ' + detail if detail else ''}")
    if not condition:
        raise AssertionError(label)


def main() -> int:
    try:
        status, health = get_json("/health")
        check("API health", status == 200 and health.get("ok") is True)
        check("App version", str(health.get("version")) == "0.8.1", str(health.get("version")))
        check(
            "Search index built",
            health.get("index_ready") is True,
            "run `python -m ingest.ingest --rebuild` first" if not health.get("index_ready") else "",
        )

        status, docs = get_json("/docs/list")
        check("Expanded demo corpus listed", status == 200 and isinstance(docs, list) and len(docs) >= 28, f"{len(docs)} PDFs")
        check(
            "Document metadata",
            all(d.get("title") and d.get("category") and d.get("owner") and d.get("url") for d in docs[:5]),
        )

        status, answer = post_json(
            "/answer",
            {
                "question": "What are users responsible for under the remote access policy?",
                "policy_only": True,
                "k": 8,
                "min_score": 0.25,
            },
        )
        check("Policy retrieval", status == 200 and answer.get("found") is True, answer.get("mode", ""))
        sources = answer.get("sources") or []
        check("Policy citations", bool(sources))
        check("Readable source titles", all((s.get("title") or "").lower() not in {"", "anonymous"} for s in sources))
        check("Source PDF links", all(s.get("url") for s in sources))
        source_titles = {str(s.get("title") or "") for s in sources}
        check(
            "Named-policy focus",
            source_titles == {"Remote Access and MFA Policy"},
            ", ".join(sorted(source_titles)),
        )

        # A section-aware policy query should return actual QC actions, never the PDF header or metadata table.
        status, qc = post_json(
            "/answer",
            {
                "question": "What checks are required before a dataset is frozen for evaluation?",
                "policy_only": True,
                "k": 8,
                "min_score": 0.25,
            },
        )
        qc_text = str(qc.get("answer") or "")
        qc_sources = qc.get("sources") or []
        check("Dataset QC retrieval", status == 200 and qc.get("found") is True, qc.get("mode", ""))
        check(
            "Dataset QC source",
            bool(qc_sources) and str(qc_sources[0].get("title") or "") == "Dataset Quality Control SOP",
            str(qc_sources[0].get("title") if qc_sources else "no source"),
        )
        check(
            "Dataset QC answer quality",
            all(term in qc_text.lower() for term in ("case count", "duplicate", "spacing", "held-out")),
            qc_text[:180],
        )
        check(
            "Answer boilerplate cleanup",
            not bool(__import__("re").search(r"(?i)\b(document type|last updated|current - synthetic demo|radoncdocs - demo corpus|page 1)\b", qc_text)),
            qc_text[:180],
        )

        # This query is exposed in the UI, so it must never dead-end in the release demo.
        status, overlay = post_json(
            "/answer",
            {
                "question": "What should I do if an imaging overlay looks misaligned?",
                "policy_only": True,
                "k": 8,
                "min_score": 0.25,
            },
        )
        overlay_text = str(overlay.get("answer") or "")
        overlay_sources = overlay.get("sources") or []
        check("Overlay troubleshooting retrieval", status == 200 and overlay.get("found") is True, overlay.get("mode", ""))
        check(
            "Overlay validation source",
            bool(overlay_sources) and str(overlay_sources[0].get("title") or "") == "Imaging Overlay Validation SOP",
            str(overlay_sources[0].get("title") if overlay_sources else "no source"),
        )
        check(
            "Overlay answer quality",
            all(term in overlay_text.lower() for term in ("spacing", "orientation", "origin", "known-point")),
            overlay_text[:180],
        )

        status, papers = get_json("/journal/list")
        check("Research library listed", status == 200 and isinstance(papers, list) and len(papers) >= 5, f"{len(papers)} papers")
        check("Research metadata", all(p.get("title") and p.get("category") and p.get("url") for p in papers[:3]))
        target = next((p for p in papers if p.get("title") == "Workflow Efficiency of Source-Cited Policy Search for Research Operations"), papers[0])
        paper = target["file"]
        status, tldr = get_json("/journal/tldr?file=" + urllib.parse.quote(paper) + "&mode=extractive")
        check("Extractive journal summary", status == 200 and bool(tldr.get("tldr")), paper)
        limitations = str(tldr.get("limitations") or "")
        check(
            "Journal text cleanup",
            not bool(__import__("re").search(r"(?i)\bpage\s+\d+\b|^\s*limitations?\b", limitations)),
            limitations[:120],
        )

        print("\nSmoke test passed. RadOncDocs v0.8.1 retrieval-quality flows are working.")
        return 0
    except (urllib.error.URLError, ConnectionError) as exc:
        print(f"[FAIL] Could not reach {BASE}: {exc}")
        print("Start the app in another terminal with `python main.py` and rerun this script.")
        return 1
    except AssertionError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
