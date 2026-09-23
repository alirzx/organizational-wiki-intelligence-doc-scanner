from fastapi import APIRouter

from app.api.v1.endpoints import extract, figure_table, health, jobs, ocr, stamp_signature, storage

router = APIRouter()
router.include_router(health.router)
router.include_router(storage.router)
router.include_router(ocr.router)
router.include_router(figure_table.router)
router.include_router(stamp_signature.router)
router.include_router(extract.router)
router.include_router(jobs.router)
