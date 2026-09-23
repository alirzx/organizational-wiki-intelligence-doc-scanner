from __future__ import annotations

import asyncio

from app.core.runtime import get_job_store
from app.jobs.celery_app import celery_app
from app.jobs.processor import process_minio_payload
from app.schemas.storage import MinioDocumentRequest


@celery_app.task(bind=True, name="wiki_hami.process_minio_document")
def process_minio_document(self, payload_data: dict, job_id: str) -> dict:
    store = get_job_store()
    payload = MinioDocumentRequest.model_validate(payload_data)
    store.update(job_id, status="processing", error=None)
    try:
        written = asyncio.run(process_minio_payload(payload))
    except Exception as exc:
        error = {"code": type(exc).__name__.upper(), "message": str(exc)}
        store.update(job_id, status="failed", error=error)
        return {"job_id": job_id, "document_id": payload.document_id, "status": "failed", "error": error}

    prefix = f"documents/{payload.document_id}"
    outputs = {
        "ocr": f"{prefix}/OCR.txt",
        "layout": None,
        "ocr_dir": f"{prefix}/OCR/",
        "figure_table_dir": f"{prefix}/Figure-Table/",
        "stamp_signature_dir": f"{prefix}/Stamp-Signature/",
    }
    store.update(job_id, status="completed", outputs=outputs, error=None)
    return {
        "job_id": job_id,
        "document_id": payload.document_id,
        "status": "completed",
        "outputs": outputs,
        "written": written,
    }
