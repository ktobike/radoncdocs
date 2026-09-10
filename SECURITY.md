# Security and data scope

RadOncDocs public demo is intentionally isolated from the internal research project it was adapted from.

## Included

- Synthetic policy and procedure PDFs created for this repository
- Synthetic research articles created for this repository
- Source code required to ingest, retrieve, summarize, and display those documents
- A manifest containing only synthetic document metadata

## Excluded

- Institutional policies, procedures, or internal documents
- Patient information or clinical source data
- Credentials, API keys, tokens, or `.env` files
- Private Chroma indexes, extracted text caches, or generated internal summaries
- Private Git history
- Institutional source documents or other private/internal assets

Administrative upload, delete, and reindex routes are disabled by default. They can only be enabled intentionally for local development with `RADONCDOCS_ENABLE_ADMIN=1`.

This repository is a portfolio demonstration, not a clinical or operational system. The included documents and research results are synthetic and should not be treated as guidance.
