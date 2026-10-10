from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SUPPORTED_MODEL_RUNTIMES = frozenset({"cpu", "gpu"})
SUPPORTED_TEXT_EXTRACTION_MODES = frozenset({"ocr", "vlm"})
SUPPORTED_OCR_BACKENDS = frozenset({"mock", "paddle", "bina_rizeh"})
SUPPORTED_VLM_BACKENDS = frozenset({"ollama"})


class Settings(BaseSettings):
    app_name: str = "Wiki Hami Extraction"
    api_prefix: str = "/api/v1"
    schema_version: str = "wiki-hami.extraction.v1"
    environment: str = "local"
    log_level: str = "INFO"
    model_runtime: str = "gpu"  # cpu | gpu; scanner Paddle/Torch dependency profile

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

    # Text extraction selection. Keep the public OCR module/API/artifact contract
    # stable while allowing the implementation to switch between classic OCR and
    # a remote VLM entirely through configuration.
    text_extraction_mode: str = "vlm"  # ocr | vlm

    # Classic OCR path.
    ocr_backend: str = "bina_rizeh"  # mock | paddle | bina_rizeh
    ocr_model_id: str = "PaddlePaddle/arabic_PP-OCRv5_mobile_rec"
    ocr_text_detection_model_name: str = "PP-OCRv5_server_det"
    ocr_device: str = "gpu:0"
    # Stability-first CPU baseline. PaddleOCR/PaddleX enable oneDNN/MKLDNN by
    # default; keep it disabled unless a target runtime has been regression-tested.
    ocr_enable_mkldnn: bool = False
    ocr_score_threshold: float = 0.45
    ocr_use_textline_orientation: bool = True
    # Bina is a line recognizer. The full-page backend pairs it with a separately
    # pinned detector and applies the official visual-to-logical text transform.
    ocr_bina_model_id: str = "Reza2kn/Bina-0.2-RizehPizeh"
    ocr_bina_revision: str = "993527413ff74ef6d446df91c715a4e0825abe5b"
    ocr_bina_score_threshold: float = 0.0
    ocr_bina_batch_size: int = 1
    ocr_detection_model_id: str = "PaddlePaddle/PP-OCRv6_medium_det"
    ocr_detection_model_revision: str = "8e0f56fb2ef86b461d99cfc7ac5c137738985f61"
    ocr_paragraph_max_gap_ratio: float = 1.8
    ocr_paragraph_min_x_overlap: float = 0.15

    # VLM path. Docker Compose owns an isolated Ollama service by default; direct
    # local processes may point this URL at any explicitly configured Ollama endpoint.
    vlm_backend: str = "ollama"
    vlm_model_id: str = "deepseek-ocr:latest"
    vlm_base_url: str = "http://localhost:11434"
    vlm_timeout_seconds: float = 360.0
    vlm_prompt: str = "\nExtract the text in the image."
    vlm_crop_margins: bool = True
    # Kept under the existing environment name for deployment compatibility.
    # When enabled, VLM extraction uses three top-to-bottom page regions first.
    vlm_region_fallback: bool = True
    vlm_classic_fallback: bool = True
    vlm_diagnostics_dir: str = "data/outputs/ocr-diagnostics"
    vlm_diagnostics_max_files: int = Field(default=64, ge=0, le=10_000)
    vlm_keep_alive: str = "30s"
    # DeepSeek-OCR supports up to 8192 tokens, but stage-safe defaults use a
    # smaller context/generation footprint. Region OCR remains the primary
    # strategy for dense pages; deployments may raise these values explicitly.
    vlm_max_tokens: int = Field(default=3072, ge=1, le=8192)
    vlm_context_size: int = Field(default=4096, ge=512, le=8192)
    vlm_repeat_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    vlm_quality_retries: int = Field(default=1, ge=0, le=2)

    # Layout localization.
    figure_table_backend: str = "mock"
    figure_table_model_id: str = "PaddlePaddle/PP-DocLayoutV3"
    figure_table_device: str = "gpu:0"
    figure_table_score_threshold: float = 0.45
    figure_labels: str = "figure,image,chart"
    table_labels: str = "table"

    # RF-DETR fine-tuned checkpoint on Hugging Face.
    stamp_signature_backend: str = "mock"
    stamp_signature_model_id: str = "bluecopa/rf-detr-stamp-signature-detector"
    stamp_signature_checkpoint_filename: str = "checkpoint_best_ema.pth"
    stamp_signature_model_revision: str = "c59fd4f451b254501700a56c7769f1a3d788c753"
    stamp_signature_device: str = "cuda:0"
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

    @field_validator("model_runtime")
    @classmethod
    def validate_model_runtime(cls, value: str) -> str:
        runtime = value.strip().lower()
        if runtime not in SUPPORTED_MODEL_RUNTIMES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_RUNTIMES))
            raise ValueError(f"model_runtime must be one of: {supported}")
        return runtime

    @field_validator("text_extraction_mode")
    @classmethod
    def validate_text_extraction_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in SUPPORTED_TEXT_EXTRACTION_MODES:
            supported = ", ".join(sorted(SUPPORTED_TEXT_EXTRACTION_MODES))
            raise ValueError(f"text_extraction_mode must be one of: {supported}")
        return mode

    @field_validator("ocr_backend")
    @classmethod
    def validate_ocr_backend(cls, value: str) -> str:
        backend = value.strip().lower()
        if backend not in SUPPORTED_OCR_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_OCR_BACKENDS))
            raise ValueError(f"ocr_backend must be one of: {supported}")
        return backend

    @field_validator("vlm_prompt")
    @classmethod
    def normalize_vlm_prompt(cls, value: str) -> str:
        # Compose env files may preserve the documented newline as literal \n.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value.replace("\\n", "\n")

    @field_validator("vlm_keep_alive")
    @classmethod
    def normalize_vlm_keep_alive(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("vlm_keep_alive must not be empty")
        return normalized

    @field_validator("vlm_backend")
    @classmethod
    def validate_vlm_backend(cls, value: str) -> str:
        backend = value.strip().lower()
        if backend not in SUPPORTED_VLM_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_VLM_BACKENDS))
            raise ValueError(f"vlm_backend must be one of: {supported}")
        return backend

    @field_validator("vlm_base_url")
    @classmethod
    def normalize_vlm_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("vlm_base_url must start with http:// or https://")
        return normalized


    @model_validator(mode="after")
    def validate_cpu_runtime_devices(self) -> "Settings":
        if self.model_runtime != "cpu":
            return self

        conflicts: list[str] = []
        if (
            self.text_extraction_mode == "ocr"
            and self.ocr_backend != "mock"
            and self.ocr_device.strip().lower() != "cpu"
        ):
            conflicts.append("WIKI_HAMI_OCR_DEVICE")
        if (
            self.figure_table_backend != "mock"
            and self.figure_table_device.strip().lower() != "cpu"
        ):
            conflicts.append("WIKI_HAMI_FIGURE_TABLE_DEVICE")
        if (
            self.stamp_signature_backend != "mock"
            and self.stamp_signature_device.strip().lower() != "cpu"
        ):
            conflicts.append("WIKI_HAMI_STAMP_SIGNATURE_DEVICE")

        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError(
                "WIKI_HAMI_MODEL_RUNTIME=cpu requires active in-process model devices "
                f"to be cpu; update: {names}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
