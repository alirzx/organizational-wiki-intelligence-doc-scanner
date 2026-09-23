from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import streamlit as st

from ui.artifacts import build_run_zip, canonical_json_bytes
from ui.visualizer import annotate_page, legend_html, open_source_image


API_BASE = os.getenv("WIKI_HAMI_API_BASE", "http://localhost:8000/api/v1").rstrip("/")
REQUEST_TIMEOUT = 300

st.set_page_config(
    page_title="Wiki Hami · Extraction Inspector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_get(path: str, **kwargs):
    return requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)


def api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT, **kwargs)


def safe_response(response: requests.Response) -> dict:
    if response.ok:
        return response.json()
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    raise RuntimeError(f"API {response.status_code}: {detail}")


def parse_metadata(raw: str) -> dict[str, Any]:
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("Document metadata must be a JSON object")
    return value


def fetch_minio_source(object_key: str) -> bytes:
    response = api_get("/storage/minio/object", params={"object_key": object_key})
    if not response.ok:
        raise RuntimeError(f"Could not load MinIO source {object_key}: {response.text}")
    return response.content


def store_run(result: dict, sources: list[dict[str, Any]], persisted: bool) -> None:
    st.session_state["extraction_run"] = {
        "result": result,
        "sources": sources,
        "persisted": persisted,
    }


with st.sidebar:
    st.title("Wiki Hami")
    st.caption("Extraction V1 · engineering inspector")
    st.code(API_BASE, language=None)
    try:
        health_response = api_get("/health")
        health = safe_response(health_response)
        st.success("API connected")
        modules = health.get("modules", {})
        st.caption(" · ".join(f"{name}: {cfg.get('backend', '?')}" for name, cfg in modules.items()))
    except Exception as exc:
        st.error(f"API unavailable: {exc}")
    st.divider()
    st.caption("Production: POST /extract/minio. Module APIs and detailed extraction paths remain engineering/debug surfaces.")

st.title("Document Extraction Inspector")
st.caption(
    "Local upload and MinIO inputs share the same preprocessing/model orchestration. "
    "Enable product artifact persistence to also write the production OCR, Figure-Table and Stamp-Signature outputs to MinIO."
)

input_tab, result_tab, api_tab = st.tabs(["Input & Run", "Results", "Integration"])

with input_tab:
    left, right = st.columns([1, 1.7])
    with left:
        document_id = st.text_input("Document ID", value="doc_demo_001")
    with right:
        document_metadata = st.text_area(
            "Document metadata JSON",
            value='{"source":"inspection"}',
            height=80,
        )

    persist_outputs = st.checkbox(
        "Persist product artifacts to MinIO",
        value=False,
        help=(
            "When enabled, the inspection flow also writes documents/<document_id>/OCR/, "
            "Figure-Table/, Stamp-Signature/ and OCR.txt using the same publisher as the product endpoint."
        ),
    )
    source_mode = st.radio("Image source", ["Local upload", "MinIO"], horizontal=True)

    if source_mode == "Local upload":
        files = st.file_uploader(
            "Choose document page images in order",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if files:
            previews = st.columns(min(4, len(files)))
            for index, uploaded in enumerate(files[:4]):
                with previews[index % len(previews)]:
                    st.image(uploaded.getvalue(), caption=f"{index + 1}. {uploaded.name}", use_container_width=True)

        if st.button("Run local extraction", type="primary", disabled=not files):
            try:
                metadata = parse_metadata(document_metadata)
                uploads = [
                    {"name": item.name, "type": item.type, "data": item.getvalue()}
                    for item in files
                ]
                multipart = [
                    ("images", (item["name"], item["data"], item["type"] or "application/octet-stream"))
                    for item in uploads
                ]
                descriptors = [
                    {
                        "page_number": index,
                        "page_id": f"{document_id.strip()}:p{index}",
                        "filename": item["name"],
                        "metadata": {"source": "streamlit_upload"},
                    }
                    for index, item in enumerate(uploads, start=1)
                ]
                form = {
                    "document_id": document_id.strip(),
                    "document_metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "pages_metadata_json": json.dumps(descriptors, ensure_ascii=False),
                    "persist_outputs": str(persist_outputs).lower(),
                }
                with st.spinner("Running all three extraction modules..."):
                    result = safe_response(api_post("/extract", files=multipart, data=form))
                store_run(
                    result,
                    [{"kind": "upload", "name": item["name"], "data": item["data"]} for item in uploads],
                    persist_outputs,
                )
                st.success("Extraction complete. Open Results.")
            except Exception as exc:
                st.error(str(exc))

    else:
        try:
            minio = safe_response(api_get("/storage/minio/health"))
            connected = bool(minio.get("connected"))
        except Exception as exc:
            connected = False
            minio = {}
            st.error(str(exc))

        cols = st.columns(3)
        cols[0].metric("MinIO", "Connected" if connected else "Unavailable")
        cols[1].metric("Endpoint", minio.get("endpoint", "—"))
        cols[2].metric("Bucket", minio.get("bucket", "—"))

        prefix = st.text_input("Object prefix", value=f"documents/{document_id.strip()}/images/")
        if connected and (st.button("Refresh objects") or "minio_objects" not in st.session_state):
            try:
                body = safe_response(api_get("/storage/minio/objects", params={"prefix": prefix, "limit": 500}))
                st.session_state["minio_objects"] = body.get("objects", [])
            except Exception as exc:
                st.error(str(exc))

        objects = st.session_state.get("minio_objects", []) if connected else []
        image_objects = [
            obj for obj in objects
            if obj.get("object_key", "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        selected_keys = st.multiselect(
            "Select pages",
            [obj["object_key"] for obj in image_objects],
            disabled=not connected,
        )
        selected = [obj for obj in image_objects if obj["object_key"] in selected_keys]

        if selected:
            st.dataframe(
                [
                    {
                        "page": index,
                        "object_key": item["object_key"],
                        "size": item.get("size"),
                        "etag": item.get("etag"),
                    }
                    for index, item in enumerate(selected, start=1)
                ],
                hide_index=True,
                use_container_width=True,
            )
            preview_key = st.selectbox("Preview source", selected_keys)
            if preview_key:
                try:
                    st.image(fetch_minio_source(preview_key), caption=preview_key, use_container_width=True)
                except Exception as exc:
                    st.warning(str(exc))

        if st.button("Run MinIO extraction", type="primary", disabled=not selected):
            try:
                metadata = parse_metadata(document_metadata)
                payload = {
                    "document_id": document_id.strip(),
                    "document_metadata": {**metadata, "source": "minio"},
                    "pages": [
                        {
                            "image_url": item["image_url"],
                            "page_number": index,
                            "page_id": f"{document_id.strip()}:p{index}",
                            "filename": item["object_key"].rsplit("/", 1)[-1],
                            "metadata": {"minio_object_key": item["object_key"]},
                        }
                        for index, item in enumerate(selected, start=1)
                    ],
                }
                with st.spinner("Reading MinIO pages and running all three modules..."):
                    result = safe_response(
                        api_post(
                            "/extract/minio/inspect",
                            params={"persist_outputs": str(persist_outputs).lower()},
                            json=payload,
                        )
                    )
                sources = [
                    {
                        "kind": "minio",
                        "name": item["object_key"],
                        "object_key": item["object_key"],
                        "data": fetch_minio_source(item["object_key"]),
                    }
                    for item in selected
                ]
                store_run(result, sources, persist_outputs)
                st.success("Extraction complete. Open Results.")
            except Exception as exc:
                st.error(str(exc))

with result_tab:
    run = st.session_state.get("extraction_run")
    if not run:
        st.info("Run an extraction first.")
    else:
        result = run["result"]
        sources = run["sources"]
        pages = result.get("pages", [])
        state = result.get("processing", {}).get("state", "unknown")
        if state == "success":
            st.success("All modules succeeded." + (" Product artifacts were persisted to MinIO." if run["persisted"] else ""))
        elif state == "partial_success":
            st.warning("Partial success. Inspect per-module status.")
        else:
            st.error("Extraction failed.")

        counts = result.get("object_counts", {})
        metrics = st.columns(5)
        metrics[0].metric("Pages", result.get("page_count", 0))
        metrics[1].metric("Paragraphs", counts.get("paragraph", 0))
        metrics[2].metric("Tables/Figures", counts.get("table", 0) + counts.get("figure", 0))
        metrics[3].metric("Stamps/Signatures", counts.get("stamp", 0) + counts.get("signature", 0))
        metrics[4].metric("Duration", f"{result.get('processing', {}).get('duration_ms', 0) / 1000:.2f}s")

        source_images = [open_source_image(source["data"]) for source in sources]
        annotated_pages = [
            (page, annotate_page(source, page.get("objects", [])))
            for page, source in zip(pages, source_images, strict=True)
        ]

        filename_root = "".join(c if c.isalnum() or c in "._-" else "_" for c in result["document_id"])
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download canonical JSON",
            canonical_json_bytes(result),
            file_name=f"{filename_root}_extraction.json",
            mime="application/json",
            use_container_width=True,
        )
        d2.download_button(
            "Download JSON + annotated pages",
            build_run_zip(result, annotated_pages),
            file_name=f"{filename_root}_extraction.zip",
            mime="application/zip",
            use_container_width=True,
        )

        if pages:
            st.markdown(legend_html(), unsafe_allow_html=True)
            page_index = st.selectbox(
                "Inspect page",
                range(len(pages)),
                format_func=lambda i: f"{pages[i]['page_number']}. {pages[i]['page_id']}",
            )
            page = pages[page_index]
            objects = page.get("objects", [])
            c1, c2 = st.columns(2)
            c1.image(source_images[page_index], caption="Source", use_container_width=True)
            c2.image(annotated_pages[page_index][1], caption="Detections", use_container_width=True)

            st.markdown("#### Module status")
            st.dataframe(
                [
                    {
                        "module": name,
                        "state": status.get("state"),
                        "duration_ms": round(status.get("duration_ms", 0), 2),
                        "backend": status.get("backend"),
                        "warning": ", ".join(status.get("warnings") or []),
                        "error": status.get("error"),
                    }
                    for name, status in page.get("modules", {}).items()
                ],
                hide_index=True,
                use_container_width=True,
            )

            tabs = st.tabs(["OCR", "Figures & Tables", "Stamps & Signatures", "Page JSON", "Document JSON"])
            with tabs[0]:
                paragraphs = [obj for obj in objects if obj.get("type") == "paragraph"]
                st.text("\n\n".join(obj.get("raw_text") or obj.get("text") or "" for obj in paragraphs))
                st.json(paragraphs)
            with tabs[1]:
                st.json([obj for obj in objects if obj.get("type") in {"figure", "table"}])
            with tabs[2]:
                st.json([obj for obj in objects if obj.get("type") in {"stamp", "signature"}])
            with tabs[3]:
                st.json(page)
            with tabs[4]:
                st.json(result)

with api_tab:
    st.subheader("Backend ↔ AI product contract")
    st.code(
        '''POST /api/v1/extract/minio
{
  "document_id": "123",
  "document_metadata": {"source": "minio"},
  "pages": [
    {
      "image_url": "http://minio:9000/media/documents/123/images/page-001.jpg",
      "page_id": "123:p1",
      "page_number": 1
    }
  ]
}

200
{
  "document_id": "123",
  "status": "success"
}''',
        language="json",
    )
    st.markdown(
        "The product endpoint persists AI-owned outputs under `documents/<document_id>/OCR/`, "
        "`Figure-Table/`, `Stamp-Signature/` plus document-level `OCR.txt`. "
        "`/ocr`, `/figure-table`, `/stamp-signature`, `/extract`, and `/extract/minio/inspect` are retained for engineering/local inspection."
    )
