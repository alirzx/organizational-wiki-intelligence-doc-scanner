# Wiki Hami Documentation

This directory is the operational and integration reference for **Extraction V1**. Documentation here describes the current asynchronous production architecture on `main`.

## Read in this order

1. [architecture.md](architecture.md) — system boundaries and component ownership
2. [workflows.md](workflows.md) — end-to-end production, debug, retry, and failure flows
3. [backend-async-contract.md](backend-async-contract.md) — authoritative Backend ↔ AI request/job/callback contract
4. [minio.md](minio.md) — bucket ownership, artifact paths, and `layout.json` v2
5. [api.md](api.md) — endpoint reference
6. [contracts.md](contracts.md) — canonical identity, geometry, status, and persistence rules
7. [models.md](models.md) — model backends, normalization boundaries, caches, and limitations
8. [deployment.md](deployment.md) — production topology, configuration, rollout, and verification

## Production invariants

These rules should remain stable unless the Backend and AI teams explicitly version the integration contract:

- Backend owns `original.pdf`, `main.txt`, and `images/`.
- AI owns `OCR/`, `Figure-Table/`, `Stamp-Signature/`, `OCR.txt`, and `layout.json`.
- Production submission is asynchronous: `POST /api/v1/extract/minio` returns `202 queued`.
- Durable AI states are `queued`, `processing`, `completed`, and `failed`.
- Backend authenticates to AI with `X-API-Key`.
- AI authenticates terminal callbacks with `Authorization: Bearer ...`.
- Successful callback payload uses top-level `result`, not `outputs`.
- The AI job-status API uses `outputs` for its internal/recovery representation.
- `layout.json` is `wiki-hami.layout.v2` and contains all five canonical object types per page.
- Public geometry is in `exif_corrected_source_pixels`.
- MinIO callback/job paths are object keys inside bucket `media`; they do not include a leading `media/` component.

## Scope

Extraction V1 produces canonical detections and storage artifacts. It does not perform template construction, semantic section inference, document linking, RAG, table-cell extraction, signature identity, stamp interpretation, or final wiki generation.
