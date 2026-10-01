from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Wiki Hami Extraction"
    api_prefix: str = "/api/v1"
    schema_version: str = "wiki-hami.extraction.v1"
    environment: str = "local"
    log_level: str = "INFO"

    max_upload_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    max_pages_per_document: int = 200
    preprocess_max_long_edge: int = 2500
    page_concurrency: int = 4
    module_timeout_seconds: float = 360.0

    # MinIO / S3 acquisition + AI artifact persistence.
    minio_enabled: bool = False
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_bucket: str = "media"
    minio_public_base_url: str = "http://localhost:9000"
    minio_browser_enabled: bool = True
    minio_list_limit: int = 500

    # Async production jobs. Redis is used as the durable job-state store even
    # when Celery's broker is switched to RabbitMQ.
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    celery_queue: str = "wiki_hami_extraction"
    job_store_backend: str = "redis"  # redis | memory (memory is test/dev only)
    job_store_redis_url: str = "redis://redis:6379/2"
    job_store_ttl_seconds: int = 7 * 24 * 60 * 60

    # Backend -> AI request authentication and AI -> Backend callback delivery.
    # Empty values keep local engineering workflows usable; production should
    # configure all three values.
    backend_api_key: str = ""
    callback_url: str = ""
    callback_token: str = ""
    callback_timeout_seconds: float = 15.0
    callback_max_attempts: int = 3

    # OCR: PaddleOCR full OCR pipeline.
    ocr_backend: str = "mock"
    ocr_model_id: str = "PaddlePaddle/arabic_PP-OCRv5_mobile_rec"
    ocr_text_detection_model_name: str = "PP-OCRv5_server_det"
    ocr_device: str = "cpu"
    # Stability-first CPU baseline. PaddleOCR/PaddleX enable oneDNN/MKLDNN by
    # default; keep it disabled unless a target runtime has been regression-tested.
    ocr_enable_mkldnn: bool = False
    ocr_score_threshold: float = 0.45
    ocr_use_textline_orientation: bool = True
    ocr_paragraph_max_gap_ratio: float = 1.8
    ocr_paragraph_min_x_overlap: float = 0.15

    # Document-scoped content-integrity grouping. Disabled by default so existing
    # deployments retain the V1 heuristic paragraph contract until explicitly
    # enabled with a validated model package.
    grouping_enabled: bool = False
    grouping_model_path: str = "models/grouping/current"
    grouping_merge_threshold: float = Field(default=0.75, ge=0, le=1)
    grouping_uncertain_lower: float = Field(default=0.55, ge=0, le=1)
    grouping_candidate_reading_window: int = Field(default=4, ge=1)
    grouping_candidate_cross_page_window: int = Field(default=3, ge=1)
    grouping_candidate_max_page_distance: int = Field(default=1, ge=0)
    grouping_candidate_max_pairs_per_block: int = Field(default=25, ge=1)

    semantic_features_enabled: bool = False
    semantic_failure_policy: str = "fallback"  # fallback | fail_fast
    embedding_provider: str = "ollama"
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "WIKI_HAMI_OLLAMA_BASE_URL"),
    )
    ollama_embedding_model: str = Field(
        default="embeddinggemma",
        validation_alias=AliasChoices(
            "OLLAMA_EMBEDDING_MODEL",
            "WIKI_HAMI_OLLAMA_EMBEDDING_MODEL",
        ),
    )
    ollama_embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    ollama_embedding_batch_size: int = Field(default=32, ge=1)
    ollama_embedding_dimensions: int = 768
    ollama_embedding_truncate: bool = False
    ollama_embedding_keep_alive: str = "5m"
    embedding_cache_max_entries: int = Field(default=4096, ge=1)
    embedding_prompt_profile: str = "sentence_similarity_v1"
    embedding_max_attempts: int = Field(default=2, ge=1, le=2)

    # Layout localization.
    figure_table_backend: str = "mock"
    figure_table_model_id: str = "PaddlePaddle/PP-DocLayoutV3"
    figure_table_device: str = "cpu"
    figure_table_score_threshold: float = 0.45
    figure_labels: str = "figure,image,chart"
    table_labels: str = "table"

    # RF-DETR fine-tuned checkpoint on Hugging Face.
    stamp_signature_backend: str = "mock"
    stamp_signature_model_id: str = "bluecopa/rf-detr-stamp-signature-detector"
    stamp_signature_checkpoint_filename: str = "checkpoint_best_ema.pth"
    stamp_signature_model_revision: str = "c59fd4f451b254501700a56c7769f1a3d788c753"
    stamp_signature_device: str = "cpu"
    stamp_signature_score_threshold: float = 0.50
    stamp_signature_cache_dir: str | None = None

    # Local UI / container defaults.
    api_base: str = "http://localhost:8000/api/v1"

    model_config = SettingsConfigDict(
        env_prefix="WIKI_HAMI_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def figure_label_set(self) -> set[str]:
        return {value.strip().lower() for value in self.figure_labels.split(",") if value.strip()}

    @property
    def table_label_set(self) -> set[str]:
        return {value.strip().lower() for value in self.table_labels.split(",") if value.strip()}

    @model_validator(mode="after")
    def validate_grouping_settings(self) -> "Settings":
        if self.grouping_uncertain_lower > self.grouping_merge_threshold:
            raise ValueError(
                "grouping_uncertain_lower must be <= grouping_merge_threshold"
            )
        if self.semantic_failure_policy not in {"fallback", "fail_fast"}:
            raise ValueError(
                "semantic_failure_policy must be 'fallback' or 'fail_fast'"
            )
        if self.embedding_provider != "ollama":
            raise ValueError("embedding_provider must be 'ollama' in v1")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must use http or https")
        if self.ollama_embedding_dimensions not in {128, 256, 512, 768}:
            raise ValueError(
                "ollama_embedding_dimensions must be one of 128, 256, 512, or 768"
            )
        if not self.ollama_embedding_keep_alive.strip():
            raise ValueError("ollama_embedding_keep_alive must not be empty")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
