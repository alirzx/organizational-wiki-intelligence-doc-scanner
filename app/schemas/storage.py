from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MODULE_REQUEST_EXAMPLE = {
    "document_id": "DOC-100",
    "image_url": "http://minio.example:9000/media/documents/DOC-100/images/page-001.jpg",
    "page_number": 1,
    "page_id": "DOC-100:p1",
    "page_metadata": {"source_asset_id": "asset_991"},
}

JobState = Literal["queued", "processing", "completed", "failed"]


class ModuleImageRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [MODULE_REQUEST_EXAMPLE]})
    document_id: str = Field(min_length=1, description="Logical document identifier.")
    image_url: str = Field(min_length=1)
    page_number: int = Field(default=1, ge=1)
    page_id: str | None = Field(default=None)
    page_metadata: dict[str, Any] = Field(default_factory=dict)


class MinioPageRequest(BaseModel):
    image_url: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    page_id: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MinioDocumentRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "123",
                    "document_metadata": {"source": "minio"},
                    "pages": [
                        {
                            "image_url": "http://minio:9000/media/documents/123/images/page-001.jpg",
                            "page_number": 1,
                            "page_id": "123:p1",
                        },
                        {
                            "image_url": "http://minio:9000/media/documents/123/images/page-002.jpg",
                            "page_number": 2,
                            "page_id": "123:p2",
                        },
                    ],
                }
            ]
        }
    )
    document_id: str = Field(min_length=1)
    pages: list[MinioPageRequest] = Field(min_length=1)
    document_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_deterministic_page_identity(self) -> "MinioDocumentRequest":
        document_id = self.document_id.strip()
        if not document_id or document_id in {".", ".."} or "/" in document_id or "\\" in document_id:
            raise ValueError("document_id must be a non-empty single path segment")
        self.document_id = document_id
        page_numbers: set[int] = set()
        page_ids: set[str] = set()
        for index, page in enumerate(self.pages, start=1):
            number = page.page_number or index
            page_id = page.page_id or f"{document_id}:p{number}"
            if number in page_numbers:
                raise ValueError(f"duplicate page_number: {number}")
            if page_id in page_ids:
                raise ValueError(f"duplicate page_id: {page_id}")
            page_numbers.add(number)
            page_ids.add(page_id)
        return self


class ExtractionJobResponse(BaseModel):
    """Immediate response from the asynchronous production endpoint."""
    job_id: str
    document_id: str
    status: Literal["queued"]


class JobError(BaseModel):
    code: str
    message: str


class JobOutputPaths(BaseModel):
    ocr: str | None = None
    layout: str | None = None
    ocr_dir: str | None = None
    figure_table_dir: str | None = None
    stamp_signature_dir: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: JobState
    created_at: str
    updated_at: str
    outputs: JobOutputPaths | None = None
    error: JobError | None = None
    callback_delivered: bool | None = None
    callback_error: str | None = None


class MinioHealthResponse(BaseModel):
    status: Literal["ok", "disabled"]
    enabled: bool
    connected: bool
    endpoint: str
    public_base_url: str
    bucket: str
    browser_enabled: bool


class MinioObjectInfo(BaseModel):
    object_key: str
    image_url: str
    size: int = Field(ge=0)
    etag: str | None = None
    last_modified: datetime | None = None
    content_type: str | None = None


class MinioObjectListResponse(BaseModel):
    bucket: str
    prefix: str
    count: int
    truncated: bool = False
    objects: list[MinioObjectInfo] = Field(default_factory=list)
