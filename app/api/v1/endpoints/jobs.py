from fastapi import APIRouter, Depends, HTTPException

from app.core.runtime import get_job_store
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
def get_job_status(job_id: str):
    record = get_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse.model_validate(record)
