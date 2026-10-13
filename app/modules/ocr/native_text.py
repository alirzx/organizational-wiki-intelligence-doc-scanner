"""Use native PDF text only when independent text detection confirms coverage."""

from app.modules.ocr.text_normalization import normalize_persian_ocr_text
from app.preprocessing.transforms import restore_bbox_to_source
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName
from app.schemas.storage import NativeTextPage
from app.schemas.common import BBox
from app.utils.ids import new_object_id


def covered_area(box, regions):
    """Exact rectangle-union coverage; overlapping regions cannot double count."""
    intersections = [
        (max(box.x1, r.x1), max(box.y1, r.y1), min(box.x2, r.x2), min(box.y2, r.y2))
        for r in regions
        if min(box.x2, r.x2) > max(box.x1, r.x1) and min(box.y2, r.y2) > max(box.y1, r.y1)
    ]
    xs = sorted({x for r in intersections for x in (r[0], r[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in intersections if r[0] <= left and r[2] >= right)
        end = None
        height = 0.0
        for top, bottom in intervals:
            height += max(0, bottom - max(top, end if end is not None else top))
            end = max(bottom, end if end is not None else bottom)
        area += (right - left) * height
    return area / (box.width * box.height) if box.width * box.height else 0.0


def missing_text_regions(page, native, detections, figures):
    # PDF font metrics and detector boxes differ slightly around the same ink.
    # A small pixel tolerance avoids treating border/ascender padding as new text.
    padding = max(native.width, native.height) * 0.005
    regions = [
        BBox(x1=max(0, box.x1-padding), y1=max(0, box.y1-padding),
             x2=min(native.width, box.x2+padding), y2=min(native.height, box.y2+padding))
        for box in [block.bbox for block in native.blocks] + native.ignored_regions
    ]
    regions += [obj.bbox for obj in figures if obj.type == ObjectType.FIGURE and obj.confidence >= 0.5]
    missing = []
    for detected in detections:
        box = restore_bbox_to_source(
            detected, page.transform,
            page.image_metadata.source_width, page.image_metadata.source_height,
        )
        if covered_area(box, regions) < 0.65:
            missing.append((detected, box))
    return missing


def native_text_objects(page, native_data, detections, figures):
    native = NativeTextPage.model_validate(native_data)
    if native.page_number != page.page_number or (
        native.width != page.image_metadata.source_width
        or native.height != page.image_metadata.source_height
    ):
        raise ValueError("Native PDF text does not match the source image geometry/identity")
    if missing_text_regions(page, native, detections, figures):
        return None
    return [
        DetectedObject(
            object_id=new_object_id("paragraph"),
            document_id=page.document_id,
            page_id=page.page_id,
            page_number=page.page_number,
            type=ObjectType.PARAGRAPH,
            bbox=block.bbox,
            text=normalize_persian_ocr_text(block.text),
            raw_text=block.text,
            confidence=1.0,
            metadata={
                "text_extraction_mode": "pdf_text",
                "ocr_reading_order": order,
                "confidence_source": "native_pdf_layer",
                "confidence_semantics": "source_layer_not_ocr_probability",
            },
            provenance=Provenance(module=ModuleName.OCR, backend="pdf_text", model_id="pdf-text-layer"),
        )
        for order, block in enumerate(native.blocks, start=1)
    ]
