from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Wiki Hami Extraction"
    api_prefix: str = "/api/v1"
    schema_version: str = "wiki-hami.extraction.v1"
    environment: str = "local"
    log_level: str = "INFO"

    max_upload_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 50_000_000
    max_pages_per_document: int = 100
    preprocess_max_long_edge: int = 2500
    page_concurrency: int = 4
    module_timeout_seconds: float = 180.0

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
    callback_timeout_seconds: float = 10.0
    callback_max_attempts: int = 3

    # OCR: PaddleOCR full OCR pipeline.
    ocr_backend: str = "mock"
    ocr_model_id: str = "PaddlePaddle/arabic_PP-OCRv5_mobile_rec"
    ocr_text_detection_model_name: str = "PP-OCRv5_server_det"
    ocr_device: str = "cpu"
    ocr_score_threshold: float = 0.45
    ocr_use_textline_orientation: bool = True
    ocr_paragraph_max_gap_ratio: float = 1.8
    ocr_paragraph_min_x_overlap: float = 0.15

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
