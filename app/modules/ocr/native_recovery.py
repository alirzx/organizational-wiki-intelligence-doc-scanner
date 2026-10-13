"""Recover missing PDF text regions without replacing verified native paragraphs."""

from dataclasses import replace
import re

from app.modules.ocr.deepseek_ocr_vlm_backend import (
    DeepSeekOCRVLMBackend,
    VLMQualityError,
    VLMTransportError,
)
from app.modules.ocr.native_text import covered_area, missing_text_regions, native_text_objects
from app.modules.ocr.vlm_postprocessing import clean_vlm_lines
from app.preprocessing.transforms import restore_bbox_to_source
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from app.schemas.storage import NativeTextPage
from app.utils.ids import new_object_id


def _text_key(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _matching_verifier_lines(recognized, detected):
    return [
        line for line in recognized
        if line.confidence >= 0.75 and covered_area(line.bbox, [detected]) >= 0.8
    ]


def _translate_crop_lines(lines, left: int, top: int):
    return [
        replace(
            line,
            bbox=BBox(
                x1=line.bbox.x1 + left,
                y1=line.bbox.y1 + top,
                x2=line.bbox.x2 + left,
                y2=line.bbox.y2 + top,
            ),
        )
        for line in lines
    ]


def _predict_selected_backend(backend, crop):
    lines = backend.predict(crop)
    if isinstance(backend, DeepSeekOCRVLMBackend):
        return clean_vlm_lines(lines)
    if backend.metadata.backend == "paddle":
        return [line for line in lines if line.confidence >= 0.75]
    # Bina and other configured classic backends already apply their own
    # recognition thresholds and text normalization.
    return lines


def recover_native_text(
    page,
    verifier,
    recovery_backend,
    figures,
    *,
    allow_classic_fallback: bool,
):
    """Merge verified PDF text with only the image regions missing from that layer.

    Paddle is used as an independent visual verifier because it exposes detection
    boxes. The selected OCR backend remains authoritative for recovery: DeepSeek,
    Paddle, and Bina therefore keep the same .env selection semantics in hybrid
    PDF pages as they do for ordinary image pages.
    """

    native_data = page.page_metadata["_pdf_text"]
    native = NativeTextPage.model_validate(native_data)
    objects = native_text_objects(page, native_data, [], figures)
    assert objects is not None

    verifier_lines = verifier.predict(page.processed_image)
    missing = missing_text_regions(page, native, verifier.last_detection_boxes, figures)
    warnings: list[str] = []

    native_text = "".join(obj.text or "" for obj in objects)
    native_keys = {_text_key(obj.text or "") for obj in objects if _text_key(obj.text or "")}
    recovered_keys: set[str] = set()

    for detected, source_box in missing:
        matched_verifier = _matching_verifier_lines(verifier_lines, detected)
        lines = []
        backend = recovery_backend.metadata
        page_space = False

        # If Paddle itself is the selected primary OCR, reuse its already-computed
        # full-page recognition rather than running the same model on the crop.
        if recovery_backend.metadata.backend == "paddle" and matched_verifier:
            lines = matched_verifier
            backend = verifier.metadata
            page_space = True

        left = max(0, int(detected.x1) - 8)
        top = max(0, int(detected.y1) - 8)
        right = min(page.processed_image.width, int(detected.x2) + 9)
        bottom = min(page.processed_image.height, int(detected.y2) + 9)
        crop = page.processed_image.crop((left, top, right, bottom))

        if not lines:
            try:
                lines = _predict_selected_backend(recovery_backend, crop)
                if not lines:
                    raise VLMQualityError("Missing PDF text region returned no reliable text")
            except (VLMQualityError, VLMTransportError) as exc:
                if isinstance(recovery_backend, DeepSeekOCRVLMBackend) and allow_classic_fallback:
                    # Preserve the normal VLM contract: DeepSeek is primary, Paddle
                    # is the bounded classic fallback when the VLM cannot recover.
                    if matched_verifier:
                        lines = matched_verifier
                        backend = verifier.metadata
                        page_space = True
                    else:
                        lines = [line for line in verifier.predict(crop) if line.confidence >= 0.75]
                        backend = verifier.metadata
                        warnings.append("missing_pdf_text_region_recovered_by_paddle_fallback")
                    if not lines:
                        if covered_area(source_box, native.image_regions) >= 0.65:
                            warnings.append("unreadable_embedded_image_text_preserved_as_image")
                            continue
                        raise exc
                elif covered_area(source_box, native.image_regions) >= 0.65:
                    warnings.append("unreadable_embedded_image_text_preserved_as_image")
                    continue
                else:
                    raise
            except RuntimeError:
                # A selected classic backend (notably Bina) may be unable to read
                # text inside an embedded figure. Keep the visual asset only when
                # PDF geometry independently proves that this is an image region.
                if covered_area(source_box, native.image_regions) >= 0.65:
                    warnings.append("unreadable_embedded_image_text_preserved_as_image")
                    continue
                raise

        native_has_letters = sum(c.isalpha() for c in native_text) > 30
        if native_has_letters and not re.search(r"[\u4e00-\u9fff]", native_text) and any(
            re.search(r"[\u4e00-\u9fff]", line.text) for line in lines
        ):
            if covered_area(source_box, native.image_regions) >= 0.65:
                warnings.append("unreadable_embedded_image_text_preserved_as_image")
                continue
            raise VLMQualityError("Missing PDF text region returned unexpected model script")

        if not page_space:
            lines = _translate_crop_lines(lines, left, top)

        for line in lines:
            value = line.text.strip()
            key = _text_key(value)
            if not key or key in native_keys or key in recovered_keys:
                continue

            box = restore_bbox_to_source(
                line.bbox,
                page.transform,
                page.image_metadata.source_width,
                page.image_metadata.source_height,
            )
            # A detector can extend slightly beyond native text: do not duplicate
            # the already verified source text in an overlapping OCR result.
            if covered_area(box, [block.bbox for block in native.blocks]) >= 0.5:
                continue

            objects.append(
                DetectedObject(
                    object_id=new_object_id("paragraph"),
                    document_id=page.document_id,
                    page_id=page.page_id,
                    page_number=page.page_number,
                    type=ObjectType.PARAGRAPH,
                    bbox=box,
                    text=value,
                    raw_text=line.raw_text if line.raw_text is not None else value,
                    confidence=line.confidence,
                    metadata={
                        "text_extraction_mode": "pdf_hybrid",
                        "ocr_strategy": "missing_pdf_region",
                        "recovery_backend": backend.backend,
                    },
                    provenance=Provenance(
                        module=ModuleName.OCR,
                        backend=backend.backend,
                        model_id=backend.model_id,
                        model_version=backend.model_revision,
                    ),
                )
            )
            recovered_keys.add(key)
            warnings.append("missing_pdf_text_region_recovered_by_ocr")

    original = [obj for obj in objects if obj.provenance.backend == "pdf_text"]

    def order(obj):
        if obj.provenance.backend == "pdf_text":
            return obj.metadata["ocr_reading_order"], 1, 0
        if not original:
            return 0, 0, obj.bbox.y1
        nearest = min(
            original,
            key=lambda other: (
                abs(other.bbox.y1 - obj.bbox.y1) + abs(other.bbox.x1 - obj.bbox.x1)
            ),
        )
        return (
            nearest.metadata["ocr_reading_order"],
            0 if obj.bbox.y1 < nearest.bbox.y1 else 2,
            obj.bbox.y1,
        )

    objects.sort(key=order)
    for index, obj in enumerate(objects, 1):
        obj.metadata["ocr_reading_order"] = index

    return objects, sorted(set(warnings))
