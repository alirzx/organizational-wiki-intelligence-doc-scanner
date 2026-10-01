"""Process-local service registry for reusable model, storage and workflow backends."""

from functools import lru_cache

from app.artifacts.publisher import ArtifactPublisher
from app.core.config import get_settings
from app.jobs.store import JobStore
from app.modules.figure_table.service import FigureTableService
from app.modules.ocr.service import OCRService
from app.modules.stamp_signature.service import StampSignatureService
from app.orchestration.extractor import ExtractionOrchestrator
from app.storage.minio_service import MinioStorageService
from app.text_processing.embeddings.cache import RunEmbeddingCache
from app.text_processing.embeddings.ollama_embedding import OllamaEmbedder


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
def get_job_store() -> JobStore:
    return JobStore(get_settings())


@lru_cache
def get_extraction_orchestrator() -> ExtractionOrchestrator:
    return ExtractionOrchestrator(
        get_settings(),
        ocr=get_ocr_service(),
        figure_table=get_figure_table_service(),
        stamp_signature=get_stamp_signature_service(),
        grouping_embedder=build_grouping_embedder(get_settings()),
    )


def build_grouping_embedder(settings=None) -> OllamaEmbedder | None:
    """Construct optional semantic infrastructure only when explicitly enabled."""
    settings = settings or get_settings()
    if not settings.semantic_features_enabled:
        return None
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        dimensions=settings.ollama_embedding_dimensions,
        timeout=settings.ollama_embedding_timeout_seconds,
        batch_size=settings.ollama_embedding_batch_size,
        truncate=settings.ollama_embedding_truncate,
        keep_alive=settings.ollama_embedding_keep_alive,
        max_attempts=settings.embedding_max_attempts,
        cache=RunEmbeddingCache(settings.embedding_cache_max_entries),
    )
