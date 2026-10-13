"""Recover missing PDF text regions without replacing verified native paragraphs."""

from dataclasses import replace
import re

from app.modules.ocr.deepseek_ocr_vlm_backend import DeepSeekOCRVLMBackend, VLMQualityError
from app.modules.ocr.native_text import covered_area, missing_text_regions, native_text_objects
from app.modules.ocr.vlm_postprocessing import clean_vlm_lines
from app.preprocessing.transforms import restore_bbox_to_source
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from app.schemas.storage import NativeTextPage
from app.utils.ids import new_object_id


def recover_native_text(page, fallback, primary, figures):
    native_data = page.page_metadata["_pdf_text"]
    native = NativeTextPage.model_validate(native_data)
    objects = native_text_objects(page, native_data, [], [], figures)
    recognized = fallback.predict(page.processed_image)
    missing = missing_text_regions(page, native, fallback.last_detection_boxes, figures)
    warnings = []
    known_text = re.sub(r"\s+", "", "".join(obj.text or "" for obj in objects))
    for detected, source_box in missing:
        lines = [line for line in recognized if line.confidence >= 0.75 and covered_area(line.bbox, [detected]) >= 0.8]
        backend = fallback.metadata
        if not lines:
            left = max(0, int(detected.x1) - 8)
            top = max(0, int(detected.y1) - 8)
            right = min(page.processed_image.width, int(detected.x2) + 9)
            bottom = min(page.processed_image.height, int(detected.y2) + 9)
            crop = page.processed_image.crop((left, top, right, bottom))
            try:
                if isinstance(primary, DeepSeekOCRVLMBackend):
                    lines = clean_vlm_lines(primary.predict(crop))
                    backend = primary.metadata
                else:
                    lines = fallback.predict(crop)
                    lines = [line for line in lines if line.confidence >= 0.75]
                if not lines:
                    raise VLMQualityError("Missing PDF text region returned no reliable text")
                native_has_letters = sum(c.isalpha() for c in known_text) > 30
                if native_has_letters and not re.search(r"[\u4e00-\u9fff]", known_text) and any(
                    re.search(r"[\u4e00-\u9fff]", line.text) for line in lines
                ):
                    raise VLMQualityError("Missing PDF text region returned unexpected model script")
                lines = [replace(line, bbox=BBox(
                    x1=line.bbox.x1 + left, y1=line.bbox.y1 + top,
                    x2=line.bbox.x2 + left, y2=line.bbox.y2 + top,
                )) for line in lines]
            except VLMQualityError:
                # The PDF's embedded image is also retained as a visual asset.
                # Record the unreadable image text instead of inventing a caption
                # or discarding the rest of the page's native text.
                if covered_area(source_box, native.image_regions) >= 0.65:
                    warnings.append("unreadable_embedded_image_text_preserved_as_image")
                    continue
                raise
        for line in lines:
            value = line.text.strip()
            if not value or re.sub(r"\s+", "", value) in known_text:
                continue
            box = restore_bbox_to_source(
                line.bbox, page.transform,
                page.image_metadata.source_width, page.image_metadata.source_height,
            )
            # A detector can extend slightly beyond native text: do not duplicate
            # the already verified source text in an overlapping OCR result.
            if covered_area(box, [block.bbox for block in native.blocks]) >= 0.5:
                continue
            objects.append(DetectedObject(
                object_id=new_object_id("paragraph"), document_id=page.document_id,
                page_id=page.page_id, page_number=page.page_number,
                type=ObjectType.PARAGRAPH, bbox=box, text=value,
                raw_text=line.raw_text if line.raw_text is not None else value,
                confidence=line.confidence,
                metadata={"text_extraction_mode": "pdf_hybrid", "ocr_strategy": "missing_pdf_region"},
                provenance=Provenance(module=ModuleName.OCR, backend=backend.backend, model_id=backend.model_id),
            ))
            warnings.append("missing_pdf_text_region_recovered_by_ocr")

    original = [obj for obj in objects if obj.provenance.backend == "pdf_text"]
    def order(obj):
        if obj.provenance.backend == "pdf_text":
            return obj.metadata["ocr_reading_order"], 1, 0
        if not original:
            return 0, 0, obj.bbox.y1
        nearest = min(original, key=lambda other: abs(other.bbox.y1-obj.bbox.y1) + abs(other.bbox.x1-obj.bbox.x1))
        return nearest.metadata["ocr_reading_order"], 0 if obj.bbox.y1 < nearest.bbox.y1 else 2, obj.bbox.y1
    objects.sort(key=order)
    for index, obj in enumerate(objects, 1):
        obj.metadata["ocr_reading_order"] = index
    return objects, sorted(set(warnings))
