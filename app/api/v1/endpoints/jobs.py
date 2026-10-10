from fastapi import APIRouter, Depends, HTTPException

from app.core.blocking import BlockingPool, run_blocking
from app.core.runtime import get_job_store
from app.jobs.celery_app import celery_app
from app.jobs.tasks import _deliver_terminal_callback, mark_job_cancelled
from app.schemas.storage import JobStatusResponse
from app.security import verify_backend_api_key


router = APIRouter(tags=["Async Jobs"])


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_backend_api_key)],
    summary="Get asynchronous extraction job status",
)
async def get_job_status(job_id: str):
    record = await run_blocking(BlockingPool.IO, get_job_store().get, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse.model_validate(record)


@router.delete(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(verify_backend_api_key)],
    summary="Cancel an asynchronous extraction job",
)
async def cancel_job(job_id: str):
    store = get_job_store()
    record = await run_blocking(BlockingPool.IO, store.get, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    if record["status"] == "completed":
        raise HTTPException(status_code=409, detail="completed job cannot be cancelled")
    if record["status"] == "failed":
        return JobStatusResponse.model_validate(record)

    message = "Extraction job cancelled before completion."
    record = await run_blocking(
        BlockingPool.IO, mark_job_cancelled, job_id, record["document_id"], message=message,
    )
    await run_blocking(
        BlockingPool.IO, celery_app.control.revoke, job_id, terminate=True, signal="SIGTERM",
    )
    record = await run_blocking(
        BlockingPool.IO, mark_job_cancelled, job_id, record["document_id"], message=message,
    )
    await run_blocking(
        BlockingPool.IO,
        _deliver_terminal_callback,
        job_id,
        record["document_id"],
        {
            "job_id": job_id,
            "document_id": record["document_id"],
            "status": "failed",
            "error": record["error"],
        },
    )
    return JobStatusResponse.model_validate(
        await run_blocking(BlockingPool.IO, store.get, job_id)
    )
