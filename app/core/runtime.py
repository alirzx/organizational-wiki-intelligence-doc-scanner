"""Process-local service registry for reusable model, storage and workflow backends."""

from functools import lru_cache

from app.artifacts.publisher import ArtifactPublisher
from app.core.config import get_settings
from app.modules.figure_table.service import FigureTableService
from app.modules.ocr.service import OCRService
from app.modules.stamp_signature.service import StampSignatureService
from app.orchestration.extractor import ExtractionOrchestrator
from app.storage.minio_service import MinioStorageService


@lru_cache
def get_ocr_service() -> OCRService:
    return OCRService(get_settings())


@lru_cache
def get_figure_table_service() -> FigureTableService:
    return FigureTableService(get_settings())


@lru_cache
def get_stamp_signature_service() -> StampSignatureService:
    return StampSignatureService(get_settings())


@lru_cache
def get_minio_storage_service() -> MinioStorageService:
    return MinioStorageService(get_settings())


@lru_cache
def get_artifact_publisher() -> ArtifactPublisher:
    return ArtifactPublisher(get_minio_storage_service())


@lru_cache
def get_extraction_orchestrator() -> ExtractionOrchestrator:
    """Build the orchestrator around the same services used by module endpoints."""
    return ExtractionOrchestrator(
        get_settings(),
        ocr=get_ocr_service(),
        figure_table=get_figure_table_service(),
        stamp_signature=get_stamp_signature_service(),
    )
