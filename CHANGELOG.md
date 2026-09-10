# Changelog

## v0.8.1 - Retrieval + lab-branding fix

- Restored the user-provided McBeth Lab mark in the app chrome.
- Procedural answers now inspect the full selected document after retrieval, so a REQUIREMENTS chunk cannot suppress a more useful PROCEDURE/CHECKLIST section.
- Keeps the synthetic-corpus disclosure while preserving the lab identity of the original project.

## v0.8.1 - Retrieval quality release candidate

- Reworked policy ingestion to index semantic sections instead of mixed whole-page chunks.
- Removed generated PDF chrome and metadata tables from searchable answer text while preserving structured metadata for citations.
- Added section-aware retrieval boosts for responsibilities, procedures, checks, requirements, scope, purpose, and exceptions.
- Moved document-type gating ahead of file-first routing so policy-only searches cannot be crowded out by stronger journal matches.
- Updated relevance checks to include title and section metadata, preserving strong matches after section-aware chunking.
- Preserved chunk identity for long policy sections instead of collapsing same-page section chunks.
- Added a synthetic Imaging Overlay Validation SOP so the published overlay-troubleshooting demo query has a policy-backed answer.
- Expanded procedural intent detection for checks, troubleshooting, and misalignment questions.
- Strengthened source snippets and fallback answers against PDF-header and metadata leakage.
- Filled sparse Recent Searches cards with useful suggested queries instead of leaving a large empty panel.
- Expanded the smoke test to assert dataset-QC answer content, overlay troubleshooting, source selection, and boilerplate cleanup.
- Regenerated the public demo corpus with 20 policies/procedures and 8 research articles.

## v0.7.0 - Release candidate

- Added explicit-document routing so named policy/SOP questions stay focused on the intended source.
- Cleaned journal summaries to remove page banners and repeated section headings from extracted text.
- Improved journal Methods output with actual method sentences when a study-design label alone is insufficient.
- Replaced institutional branding with an independent RadOncDocs mark and favicon.
- Reduced repetitive demo labeling in the application while keeping one clear synthetic-corpus badge.
- Regenerated all 27 demo PDFs with RadOncDocs-only branding.
- Added URL-based view/query initialization for easier deep linking and repeatable demos.
- Strengthened smoke tests for named-policy focus and journal text cleanup.
- Strengthened the safety check to prevent an institutional logo asset from shipping.
- Updated README and security documentation for the public release candidate.
