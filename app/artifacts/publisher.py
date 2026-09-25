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

MODULE_OBJECT_TYPES = {
    ModuleName.OCR: {ObjectType.PARAGRAPH},
    ModuleName.FIGURE_TABLE: {ObjectType.FIGURE, ObjectType.TABLE},
    ModuleName.STAMP_SIGNATURE: {ObjectType.STAMP, ObjectType.SIGNATURE},
}

VISUAL_OBJECT_TYPES = {
    ObjectType.FIGURE,
    ObjectType.TABLE,
    ObjectType.STAMP,
    ObjectType.SIGNATURE,
}


class ArtifactPublisher:
    """Persist deterministic AI-owned artifacts for one document."""

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

    @staticmethod
    def _crop_key_map(
        document_prefix: str,
        page_token: str,
        module_name: ModuleName,
        objects: list[DetectedObject],
    ) -> dict[str, str]:
        if module_name == ModuleName.OCR:
            return {}

        directory = MODULE_DIRS[module_name]
        allowed_types = MODULE_OBJECT_TYPES[module_name] & VISUAL_OBJECT_TYPES
        counters: dict[ObjectType, int] = {}
        crop_keys: dict[str, str] = {}
        for obj in objects:
            if obj.type not in allowed_types:
                continue
            counters[obj.type] = counters.get(obj.type, 0) + 1
            crop_keys[obj.object_id] = (
                f"{document_prefix}/{directory}/{page_token}-"
                f"{obj.type.value}-{counters[obj.type]:03d}.png"
            )
        return crop_keys

    @staticmethod
    def _reading_order_key(obj: dict) -> tuple:
        bbox = obj["bbox"]
        return (
            bbox["y1"],
            bbox["x1"],
            bbox["y2"],
            bbox["x2"],
            obj["type"],
            obj["object_id"],
        )

    @classmethod
    def _layout_document(cls, run: DocumentRunResult) -> dict:
        document_prefix = cls._document_prefix(run.response.document_id)
        pages: list[dict] = []
        document_counts: dict[str, int] = {object_type.value: 0 for object_type in ObjectType}

        for page_run in sorted(run.pages, key=lambda item: item.response.page_number):
            page_number = page_run.response.page_number
            page_token = f"page-{page_number:03d}"
            page_objects: list[dict] = []
            page_counts: dict[str, int] = {object_type.value: 0 for object_type in ObjectType}

            for module_name in (
                ModuleName.OCR,
                ModuleName.FIGURE_TABLE,
                ModuleName.STAMP_SIGNATURE,
            ):
                module_result = page_run.modules.get(module_name)
                if module_result is None:
                    continue

                directory = MODULE_DIRS[module_name]
                raw_artifact = f"{document_prefix}/{directory}/{page_token}.json.txt"
                crop_keys = cls._crop_key_map(
                    document_prefix,
                    page_token,
                    module_name,
                    module_result.objects,
                )
                allowed_types = MODULE_OBJECT_TYPES[module_name]

                for obj in module_result.objects:
                    if obj.type not in allowed_types:
                        continue

                    item = obj.model_dump(mode="json")
                    item["module"] = module_name.value
                    item["reading_order"] = 0
                    item["artifacts"] = {
                        "module_result": raw_artifact,
                        "plain_text": (
                            f"{document_prefix}/{directory}/{page_token}-text.txt"
                            if module_name == ModuleName.OCR
                            else None
                        ),
                        "crop": crop_keys.get(obj.object_id),
                    }
                    page_objects.append(item)
                    page_counts[obj.type.value] += 1
                    document_counts[obj.type.value] += 1

            page_objects.sort(key=cls._reading_order_key)
            for reading_order, obj in enumerate(page_objects, start=1):
                obj["reading_order"] = reading_order

            image = page_run.response.image.model_dump(mode="json")
            transform = page_run.response.transform.model_dump(mode="json")
            pages.append(
                {
                    "page_id": page_run.response.page_id,
                    "page_number": page_number,
                    "width": image["source_width"],
                    "height": image["source_height"],
                    "coordinate_space": image["source_coordinate_space"],
                    "image": image,
                    "transform": transform,
                    "reading_order_method": "bbox_top_to_bottom_then_left_to_right",
                    "object_count": len(page_objects),
                    "object_counts": page_counts,
                    "objects": page_objects,
                }
            )

        return {
            "schema_version": "wiki-hami.layout.v2",
            "document_id": run.response.document_id,
            "document_metadata": run.response.document_metadata,
            "page_count": len(pages),
            "object_count": sum(document_counts.values()),
            "object_counts": document_counts,
            "pages": pages,
        }

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

                crop_keys = self._crop_key_map(
                    document_prefix,
                    page_token,
                    module_name,
                    module_result.objects,
                )
                for obj in module_result.objects:
                    crop_key = crop_keys.get(obj.object_id)
                    if crop_key is None:
                        continue
                    self.storage.put_bytes(
                        crop_key,
                        self._crop_png(page_run.page.source_image, obj),
                        content_type="image/png",
                    )
                    written.append(crop_key)

        aggregate_key = f"{document_prefix}/OCR.txt"
        aggregate_text = "\n\n".join(text for text in aggregate_ocr_pages if text).strip()
        if aggregate_text:
            aggregate_text += "\n"
        self.storage.put_text(aggregate_key, aggregate_text)
        written.append(aggregate_key)

        layout_key = f"{document_prefix}/layout.json"
        self.storage.put_text(
            layout_key,
            self._json_text(self._layout_document(run)),
            content_type="application/json; charset=utf-8",
        )
        written.append(layout_key)
        return written
