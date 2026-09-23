from __future__ import annotations

import json
import math
from io import BytesIO

from PIL import Image

from app.orchestration.extractor import DocumentRunResult
from app.schemas.detection import DetectedObject, ObjectType
from app.schemas.status import ModuleName
from app.storage.minio_service import MinioStorageService


MODULE_DIRS = {
    ModuleName.OCR: "OCR",
    ModuleName.FIGURE_TABLE: "Figure-Table",
    ModuleName.STAMP_SIGNATURE: "Stamp-Signature",
}


class ArtifactPublisher:
    """Build and persist deterministic extraction artifacts for one document.

    The publisher owns only the three AI output prefixes and the document-level
    ``OCR.txt`` compatibility artifact. Backend-owned source objects such as
    ``original.pdf``, ``main.txt`` and ``images/`` are never modified.
    """

    def __init__(self, storage: MinioStorageService):
        self.storage = storage

    @staticmethod
    def _document_prefix(document_id: str) -> str:
        clean = document_id.strip().strip("/")
        if not clean or clean in {".", ".."} or "/" in clean:
            raise ValueError("document_id must be a non-empty single path segment")
        return f"documents/{clean}"

    @staticmethod
    def _json_text(value) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    @staticmethod
    def _ocr_plain_text(objects: list[DetectedObject]) -> str:
        lines = [
            (obj.raw_text or obj.text or "").strip()
            for obj in objects
            if obj.type == ObjectType.PARAGRAPH and (obj.raw_text or obj.text)
        ]
        return "\n\n".join(line for line in lines if line).strip() + ("\n" if lines else "")

    @staticmethod
    def _crop_png(source: Image.Image, obj: DetectedObject) -> bytes:
        width, height = source.size
        left = max(0, min(width - 1, math.floor(obj.bbox.x1)))
        top = max(0, min(height - 1, math.floor(obj.bbox.y1)))
        right = max(left + 1, min(width, math.ceil(obj.bbox.x2)))
        bottom = max(top + 1, min(height, math.ceil(obj.bbox.y2)))
        crop = source.crop((left, top, right, bottom))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()

    def _cleanup_owned_prefixes(self, document_prefix: str) -> None:
        for directory in MODULE_DIRS.values():
            self.storage.delete_prefix(f"{document_prefix}/{directory}/")

    def publish(self, run: DocumentRunResult) -> list[str]:
        document_prefix = self._document_prefix(run.response.document_id)
        self._cleanup_owned_prefixes(document_prefix)

        written: list[str] = []
        aggregate_ocr_pages: list[str] = []

        for page_run in run.pages:
            page_number = page_run.response.page_number
            page_token = f"page-{page_number:03d}"

            for module_name, module_result in page_run.modules.items():
                if module_result is None:
                    raise RuntimeError(
                        f"cannot publish artifacts: {module_name.value} failed on page {page_number}"
                    )

                directory = MODULE_DIRS[module_name]
                raw_key = f"{document_prefix}/{directory}/{page_token}.json.txt"
                self.storage.put_text(
                    raw_key,
                    self._json_text(module_result),
                    content_type="application/json; charset=utf-8",
                )
                written.append(raw_key)

                if module_name == ModuleName.OCR:
                    plain_text = self._ocr_plain_text(module_result.objects)
                    text_key = f"{document_prefix}/{directory}/{page_token}-text.txt"
                    self.storage.put_text(text_key, plain_text)
                    written.append(text_key)
                    aggregate_ocr_pages.append(plain_text.rstrip())
                    continue

                allowed_types = (
                    {ObjectType.FIGURE, ObjectType.TABLE}
                    if module_name == ModuleName.FIGURE_TABLE
                    else {ObjectType.STAMP, ObjectType.SIGNATURE}
                )
                counters: dict[ObjectType, int] = {}
                for obj in module_result.objects:
                    if obj.type not in allowed_types:
                        continue
                    counters[obj.type] = counters.get(obj.type, 0) + 1
                    crop_key = (
                        f"{document_prefix}/{directory}/{page_token}-"
                        f"{obj.type.value}-{counters[obj.type]:03d}.png"
                    )
                    self.storage.put_bytes(
                        crop_key,
                        self._crop_png(page_run.page.source_image, obj),
                        content_type="image/png",
                    )
                    written.append(crop_key)

        # Backend compatibility artifact documented in the integration contract.
        # Per-page OCR files remain the canonical granular outputs under OCR/.
        aggregate_key = f"{document_prefix}/OCR.txt"
        aggregate_text = "\n\n".join(text for text in aggregate_ocr_pages if text).strip()
        if aggregate_text:
            aggregate_text += "\n"
        self.storage.put_text(aggregate_key, aggregate_text)
        written.append(aggregate_key)
        return written
