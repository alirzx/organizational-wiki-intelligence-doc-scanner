from __future__ import annotations

import asyncio
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
from ui.module_runs import (
    ALL_MODULES,
    MODULE_ENDPOINTS,
    MODULE_LABELS,
    build_module_document_result,
    build_module_request,
    success_message,
)
from ui.visualizer import annotate_page, legend_html, open_source_image


API_BASE = os.getenv("WIKI_HAMI_API_BASE", "http://localhost:8000/api/v1").rstrip("/")
API_KEY = os.getenv("WIKI_HAMI_BACKEND_API_KEY", "")
REQUEST_TIMEOUT = 300

st.set_page_config(
    page_title="Wiki Hami · Extraction Inspector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def auth_headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


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


def store_run(
    result: dict,
    sources: list[dict[str, Any]],
    persisted: bool,
    executed_modules: list[str],
) -> None:
    st.session_state["extraction_run"] = {
        "result": result,
        "sources": sources,
        "persisted": persisted,
        "executed_modules": executed_modules,
    }


def build_minio_payload(document_id: str, document_metadata: str, selected: list[dict]) -> dict:
    metadata = parse_metadata(document_metadata)
    return {
        "document_id": document_id,
        "document_metadata": {**metadata, "source": "minio"},
        "pages": [
            {
                "image_url": item["image_url"],
                "page_number": index,
                "page_id": f"{document_id}:p{index}",
                "filename": item["object_key"].rsplit("/", 1)[-1],
                "metadata": {"minio_object_key": item["object_key"]},
            }
            for index, item in enumerate(selected, start=1)
        ],
    }


def execution_buttons(*, key_prefix: str, disabled: bool) -> str | None:
    choices = [
        ("all", "Run All Modules"),
        ("ocr", "Run OCR Only"),
        ("figure_table", "Run Figure/Table Only"),
        ("stamp_signature", "Run Stamp/Signature Only"),
    ]
    columns = st.columns(4)
    for column, (choice, label) in zip(columns, choices, strict=True):
        if column.button(
            label,
            type="primary" if choice == "all" else "secondary",
            disabled=disabled,
            key=f"{key_prefix}_{choice}",
            use_container_width=True,
        ):
            return choice
    return None


def run_local_module(
    module: str,
    *,
    document_id: str,
    document_metadata: dict[str, Any],
    uploads: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.core.config import get_settings
    from app.core.runtime import (
        get_figure_table_service,
        get_ocr_service,
        get_stamp_signature_service,
    )
    from app.preprocessing.pipeline import prepare_page
    from app.schemas.image import ImageSourceMetadata
    from app.utils.ids import new_request_id

    settings = get_settings()
    service = {
        "ocr": get_ocr_service,
        "figure_table": get_figure_table_service,
        "stamp_signature": get_stamp_signature_service,
    }[module]()
    pages = [
        prepare_page(
            data=item["data"],
            filename=item["name"],
            mime_type=item["type"],
            document_id=document_id,
            page_id=f"{document_id}:p{index}",
            page_number=index,
            page_metadata={"source": "streamlit_upload"},
            settings=settings,
            source=ImageSourceMetadata(type="upload"),
        )
        for index, item in enumerate(uploads, start=1)
    ]
    request_id = new_request_id()

    async def run_pages():
        return [await service.run(page, request_id) for page in pages]

    responses = asyncio.run(run_pages())
    page_metadata = {page.page_id: page.page_metadata for page in pages}
    return build_module_document_result(
        [response.model_dump(mode="json") for response in responses],
        document_metadata=document_metadata,
        page_metadata=page_metadata,
    )


def run_minio_module(
    module: str,
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    responses = [
        safe_response(api_post(MODULE_ENDPOINTS[module], json=build_module_request(payload["document_id"], page)))
        for page in payload["pages"]
    ]
    page_metadata = {page["page_id"]: page.get("metadata") or {} for page in payload["pages"]}
    return build_module_document_result(
        responses,
        document_metadata=payload["document_metadata"],
        page_metadata=page_metadata,
    )


with st.sidebar:
    st.title("Wiki Hami")
    st.caption("Extraction V1 · engineering inspector")
    st.code(API_BASE, language=None)
    try:
        health = safe_response(api_get("/health"))
        st.success("API connected")
        modules = health.get("modules", {})
        st.caption(" · ".join(f"{name}: {cfg.get('backend', '?')}" for name, cfg in modules.items()))
    except Exception as exc:
        st.error(f"API unavailable: {exc}")
    st.divider()
    st.caption("Production: POST /extract/minio returns 202 + job_id. Inspection remains synchronous.")

st.title("Document Extraction Inspector")
st.caption(
    "Local upload and MinIO inspection remain synchronous for engineering. "
    "The production MinIO action submits the same pages asynchronously through Celery."
)

input_tab, result_tab, api_tab = st.tabs(["Input & Run", "Results", "Integration"])

with input_tab:
    left, right = st.columns([1, 1.7])
    with left:
        document_id = st.text_input("Document ID", value="").strip()
    with right:
        document_metadata = st.text_area("Document metadata JSON", value='{"source":"inspection"}', height=80)

    persist_outputs = st.checkbox(
        "Persist product artifacts during synchronous inspection",
        value=False,
        help=(
            "Run All Modules only. Writes OCR/, Figure-Table/, Stamp-Signature/, "
            "OCR.txt and layout.json after a successful inspection run."
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

        local_choice = execution_buttons(
            key_prefix="local_run",
            disabled=not files or not document_id,
        )
        st.caption(
            "Run All uses POST /extract. Single-module local runs execute only the selected "
            "existing service in the Streamlit process and do not persist artifacts."
        )
        if local_choice:
            try:
                metadata = parse_metadata(document_metadata)
                uploads = [{"name": item.name, "type": item.type, "data": item.getvalue()} for item in files]
                if local_choice == "all":
                    multipart = [
                        ("images", (item["name"], item["data"], item["type"] or "application/octet-stream"))
                        for item in uploads
                    ]
                    descriptors = [
                        {
                            "page_number": index,
                            "page_id": f"{document_id}:p{index}",
                            "filename": item["name"],
                            "metadata": {"source": "streamlit_upload"},
                        }
                        for index, item in enumerate(uploads, start=1)
                    ]
                    form = {
                        "document_id": document_id,
                        "document_metadata_json": json.dumps(metadata, ensure_ascii=False),
                        "pages_metadata_json": json.dumps(descriptors, ensure_ascii=False),
                        "persist_outputs": str(persist_outputs).lower(),
                    }
                    with st.spinner("Running all three extraction modules..."):
                        result = safe_response(api_post("/extract", files=multipart, data=form))
                    executed_modules = list(ALL_MODULES)
                    persisted = persist_outputs
                else:
                    with st.spinner(f"Running {MODULE_LABELS[local_choice]} only..."):
                        result = run_local_module(
                            local_choice,
                            document_id=document_id,
                            document_metadata=metadata,
                            uploads=uploads,
                        )
                    executed_modules = [local_choice]
                    persisted = False
                    if persist_outputs:
                        st.warning(
                            "Artifact persistence remains available only for Run All Modules; "
                            "this single-module inspection was not persisted."
                        )
                store_run(
                    result,
                    [{"kind": "upload", "name": item["name"], "data": item["data"]} for item in uploads],
                    persisted,
                    executed_modules,
                )
                st.success("Inspection complete. Open Results.")
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

        prefix = f"documents/{document_id}/images/" if document_id else "documents/"
        st.text_input("Object prefix", value=prefix, disabled=True, key=f"prefix_{document_id or 'root'}")
        refresh = st.button("Refresh objects", disabled=not connected or not document_id)
        cache_key = f"minio_objects:{document_id}"
        if connected and document_id and (refresh or cache_key not in st.session_state):
            try:
                body = safe_response(api_get("/storage/minio/objects", params={"prefix": prefix, "limit": 500}))
                st.session_state[cache_key] = body.get("objects", [])
            except Exception as exc:
                st.error(str(exc))

        objects = st.session_state.get(cache_key, []) if connected and document_id else []
        image_objects = [obj for obj in objects if obj.get("object_key", "").lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        selected_keys = st.multiselect(
            "Select pages",
            [obj["object_key"] for obj in image_objects],
            disabled=not connected or not document_id,
        )
        selected = [obj for obj in image_objects if obj["object_key"] in selected_keys]

        if selected:
            st.dataframe(
                [{"page": i, "object_key": item["object_key"], "size": item.get("size"), "etag": item.get("etag")} for i, item in enumerate(selected, 1)],
                hide_index=True,
                use_container_width=True,
            )
            preview_key = st.selectbox("Preview source", selected_keys)
            if preview_key:
                try:
                    st.image(fetch_minio_source(preview_key), caption=preview_key, use_container_width=True)
                except Exception as exc:
                    st.warning(str(exc))

        minio_choice = execution_buttons(
            key_prefix="minio_run",
            disabled=not selected or not document_id,
        )
        st.caption(
            "Run All uses /extract/minio/inspect. Single-module runs call the existing "
            "per-page module endpoint and do not persist artifacts."
        )
        product_clicked = st.button(
            "Submit production async job",
            disabled=not selected or not document_id,
            use_container_width=True,
        )

        if minio_choice:
            try:
                payload = build_minio_payload(document_id, document_metadata, selected)
                if minio_choice == "all":
                    with st.spinner("Running all three modules synchronously..."):
                        result = safe_response(api_post(
                            "/extract/minio/inspect",
                            params={"persist_outputs": str(persist_outputs).lower()},
                            json=payload,
                        ))
                    executed_modules = list(ALL_MODULES)
                    persisted = persist_outputs
                else:
                    with st.spinner(f"Running {MODULE_LABELS[minio_choice]} only..."):
                        result = run_minio_module(minio_choice, payload=payload)
                    executed_modules = [minio_choice]
                    persisted = False
                    if persist_outputs:
                        st.warning(
                            "Artifact persistence remains available only for Run All Modules; "
                            "this single-module inspection was not persisted."
                        )
                sources = [
                    {"kind": "minio", "name": item["object_key"], "object_key": item["object_key"], "data": fetch_minio_source(item["object_key"])}
                    for item in selected
                ]
                store_run(result, sources, persisted, executed_modules)
                st.success("Inspection complete. Open Results.")
            except Exception as exc:
                st.error(str(exc))

        if product_clicked:
            try:
                payload = build_minio_payload(document_id, document_metadata, selected)
                accepted = safe_response(api_post("/extract/minio", json=payload, headers=auth_headers()))
                st.session_state["async_job"] = accepted
                st.success(f"Accepted: {accepted['job_id']}")
            except Exception as exc:
                st.error(str(exc))

        job = st.session_state.get("async_job")
        if job:
            st.markdown("#### Production async job")
            st.json(job)
            if st.button("Refresh job status"):
                try:
                    status_body = safe_response(api_get(f"/jobs/{job['job_id']}", headers=auth_headers()))
                    st.session_state["async_job"] = status_body
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

with result_tab:
    run = st.session_state.get("extraction_run")
    if not run:
        st.info("Run a synchronous extraction first.")
    else:
        result = run["result"]
        sources = run["sources"]
        executed_modules = run.get("executed_modules", list(ALL_MODULES))
        pages = result.get("pages", [])
        state = result.get("processing", {}).get("state", "unknown")
        if state == "success":
            st.success(success_message(executed_modules, persisted=run["persisted"]))
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
        annotated_pages = [(page, annotate_page(source, page.get("objects", []))) for page, source in zip(pages, source_images, strict=True)]
        filename_root = "".join(c if c.isalnum() or c in "._-" else "_" for c in result["document_id"])
        d1, d2 = st.columns(2)
        d1.download_button("Download canonical JSON", canonical_json_bytes(result), file_name=f"{filename_root}_extraction.json", mime="application/json", use_container_width=True)
        d2.download_button("Download JSON + annotated pages", build_run_zip(result, annotated_pages), file_name=f"{filename_root}_extraction.zip", mime="application/zip", use_container_width=True)

        if pages:
            st.markdown(legend_html(), unsafe_allow_html=True)
            page_index = st.selectbox("Inspect page", range(len(pages)), format_func=lambda i: f"{pages[i]['page_number']}. {pages[i]['page_id']}")
            page = pages[page_index]
            objects = page.get("objects", [])
            c1, c2 = st.columns(2)
            c1.image(source_images[page_index], caption="Source", use_container_width=True)
            c2.image(annotated_pages[page_index][1], caption="Detections", use_container_width=True)
            st.markdown("#### Module status")
            st.dataframe([
                {
                    "module": name,
                    "state": module_status.get("state"),
                    "duration_ms": round(module_status.get("duration_ms", 0), 2),
                    "backend": module_status.get("backend"),
                    "warning": ", ".join(module_status.get("warnings") or []),
                    "error": module_status.get("error"),
                }
                for name, module_status in page.get("modules", {}).items()
            ], hide_index=True, use_container_width=True)
            tabs = st.tabs(["OCR", "Figures & Tables", "Stamps & Signatures", "Page JSON", "Document JSON"])
            with tabs[0]:
                if "ocr" not in page.get("modules", {}):
                    st.info("OCR was not executed for this run.")
                else:
                    paragraphs = [obj for obj in objects if obj.get("type") == "paragraph"]
                    st.text("\n\n".join(obj.get("raw_text") or obj.get("text") or "" for obj in paragraphs))
                    st.json(paragraphs)
            with tabs[1]:
                if "figure_table" not in page.get("modules", {}):
                    st.info("Figure/Table was not executed for this run.")
                else:
                    st.json([obj for obj in objects if obj.get("type") in {"figure", "table"}])
            with tabs[2]:
                if "stamp_signature" not in page.get("modules", {}):
                    st.info("Stamp/Signature was not executed for this run.")
                else:
                    st.json([obj for obj in objects if obj.get("type") in {"stamp", "signature"}])
            with tabs[3]:
                st.json(page)
            with tabs[4]:
                st.json(result)

with api_tab:
    st.subheader("Backend ↔ AI asynchronous product contract")
    st.code('''POST /api/v1/extract/minio\nX-API-Key: <shared key>\n\n{\n  "document_id": "41",\n  "document_metadata": {"source": "minio"},\n  "pages": [{\n    "image_url": "http://minio:9000/media/documents/41/images/page-001.jpg",\n    "page_id": "41:p1",\n    "page_number": 1\n  }]\n}\n\n202 Accepted\n{\n  "job_id": "...",\n  "document_id": "41",\n  "status": "queued"\n}\n\nGET /api/v1/jobs/{job_id}\n\nTerminal callback -> Backend\n{\n  "job_id": "...",\n  "document_id": "41",\n  "status": "completed",\n  "result": {\n    "ocr": "documents/41/OCR.txt",\n    "layout": "documents/41/layout.json"\n  }\n}''', language="json")
    st.markdown(
        "Production outputs are written under `documents/<document_id>/`: per-page module artifacts, "
        "document-level `OCR.txt`, and `layout.json`. `main.txt` remains Backend-owned and is never modified."
    )
