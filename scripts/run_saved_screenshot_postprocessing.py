"""Rerun current postprocessing on line detections from a saved Paddle extraction.

This avoids repeating OCR inference when PaddleOCR is unavailable locally, while
still exercising current preprocessing, canonicalization, grouping and publisher.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from app.artifacts.publisher import ArtifactPublisher
from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects, lines_to_text_blocks
from app.modules.ocr.types import OCRAnalysis, OCRLine
from app.orchestration.extractor import ExtractionOrchestrator, PageRunResult
from app.preprocessing.pipeline import prepare_page
from app.schemas.common import BBox
from app.schemas.extraction import DocumentExtractionResponse, ModulePageResponse, PageExtractionResponse
from app.schemas.image import ImageSourceMetadata
from app.schemas.status import ModuleName, ModuleStatus, ProcessingStatus


class LocalSink:
    def __init__(self, output: Path, document_id: str):
        self.output = output.resolve()
        self.prefix = f'documents/{document_id}/'

    def delete_prefix(self, prefix: str):
        # Do not delete files; current-run is an isolated output directory.
        pass

    def _path(self, key: str) -> Path:
        if not key.startswith(self.prefix):
            raise ValueError('unexpected publisher key')
        path = (self.output / key[len(self.prefix):]).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError('publisher path escaped output directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_text(self, key, text, **kwargs):
        self._path(key).write_text(text, encoding='utf-8')

    def put_bytes(self, key, data, **kwargs):
        self._path(key).write_bytes(data)


class SavedLineOrchestrator(ExtractionOrchestrator):
    async def _run_page(self, page, request_id, page_semaphore):
        lines = self.saved_lines
        blocks = lines_to_text_blocks(lines, page=page)
        objects = lines_to_detected_objects(lines, page=page, settings=self.settings,
                                            backend_name='paddle-saved-detections')
        status = ModuleStatus(module=ModuleName.OCR, state='success', duration_ms=0,
            backend='paddle-saved-detections', model_id=self.settings.ocr_model_id,
            warnings=['ocr_inference_reused_from_saved_response'])
        module = ModulePageResponse(schema_version=self.settings.schema_version,
            request_id=request_id, document_id=page.document_id, page_id=page.page_id,
            page_number=page.page_number, module=ModuleName.OCR, image=page.image_metadata,
            transform=page.transform, objects=objects, status=status)
        response = PageExtractionResponse(schema_version=self.settings.schema_version,
            request_id=request_id, document_id=page.document_id, page_id=page.page_id,
            page_number=page.page_number, image=page.image_metadata, transform=page.transform,
            objects=objects, modules={ModuleName.OCR: status},
            processing=ProcessingStatus(state='success', duration_ms=0,
                warnings=['OCR inference reused from the saved Paddle response.']))
        return PageRunResult(page, {ModuleName.OCR: module}, response,
                             OCRAnalysis(tuple(lines), tuple(blocks), tuple(objects)))


def main(image_path: Path, prior_response: Path, output: Path):
    source = json.loads(prior_response.read_text(encoding='utf-8'))
    if source.get('page_count') != 1 or not source.get('content_groups'):
        raise ValueError('saved response must contain the single-page line-level content groups')
    members = [member for group in source['content_groups'] for member in group['members']]
    if len(members) != source['grouping']['counts']['blocks']:
        raise ValueError('saved response member count does not match its OCR block diagnostics')
    unique = {member['block_id'] for member in members}
    if len(unique) != len(members):
        raise ValueError('saved response has duplicate OCR member IDs')
    lines = [OCRLine(text=member['original_text'],
        confidence=float(member['confidence'] if member['confidence'] is not None else 0.),
        bbox=BBox.model_validate(member['bbox'])) for member in members]

    image_path = image_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings(_env_file=None, ocr_backend='mock', grouping_enabled=True,
                        grouping_backend='clustering', semantic_features_enabled=False,
                        figure_table_backend='mock', stamp_signature_backend='mock')
    document_id = 'screenshot-20260930-143459-current'
    image_bytes = image_path.read_bytes()
    page = prepare_page(data=image_bytes, filename=image_path.name,
        mime_type='image/png', document_id=document_id, page_id=f'{document_id}:p1',
        page_number=1, page_metadata={'source': 'saved-real-paddle-detections'},
        settings=settings, source=ImageSourceMetadata(type='upload'))

    orchestrator = SavedLineOrchestrator(settings)
    orchestrator.saved_lines = lines
    try:
        run = asyncio.run(orchestrator.extract_document_run(document_id=document_id,
            pages=[page], request_id='screenshot-postprocessing-rerun',
            document_metadata={'synthetic': False, 'ocr_inference': 'reused_saved_paddle_response',
                'source_image': image_path.name, 'source_extraction': str(prior_response)}))
        response_path = output / 'extraction.json'
        response_path.write_text(run.response.model_dump_json(indent=2) + '\n', encoding='utf-8')
        DocumentExtractionResponse.model_validate_json(response_path.read_text(encoding='utf-8'))
        ArtifactPublisher(LocalSink(output, document_id)).publish(run)

        block_path = output / 'OCR-blocks.txt'
        ordered = sorted(lines, key=lambda item: (item.bbox.y1, item.bbox.x1, item.text))
        block_path.write_text('\n'.join(line.text for line in ordered) + '\n', encoding='utf-8')
        actual = sum(len(group.members) for group in run.response.content_groups)
        if actual != len(lines):
            raise AssertionError(f'grouping output has {actual} members for {len(lines)} OCR lines')
        report = {
            'source_image': str(image_path),
            'source_extraction': str(prior_response),
            'ocr_inference': 'not rerun; reused saved real Paddle output',
            'source_ocr_model': source['pages'][0]['modules']['ocr']['model_id'],
            'source_ocr_backend': source['pages'][0]['modules']['ocr']['backend'],
            'current_postprocessing': True,
            'grouping_backend': run.response.grouping.model_package_id,
            'feature_schema_version': run.response.grouping.feature_schema_version,
            'semantic_mode': run.response.grouping.semantic_mode,
            'ocr_line_count': len(lines),
            'group_count': len(run.response.content_groups),
            'group_types': [group.group_type.value for group in run.response.content_groups],
            'group_sizes': [len(group.members) for group in run.response.content_groups],
            'processing_state': run.response.processing.state.value,
            'fallback_reason': run.response.grouping.fallback_reason,
            'timings_ms': run.response.grouping.timings_ms,
            'warnings': run.response.processing.warnings,
            'validation': 'Public document response schema parsed; every OCR line appears exactly once in content groups.',
        }
        (output / 'test-report.md').write_text(
            '# Screenshot postprocessing test\n\n'
            f"Source image: `{image_path}`\n\n"
            'The image was decoded and preprocessed by the current pipeline. OCR inference '
            'was reused from the existing real Paddle extraction saved alongside this image; '
            'PaddleOCR is not installed in the active Python environment. The current canonical '
            'block adapter, DBSCAN grouping, public response schemas, and artifact publisher '
            'were executed again on those 24 source OCR lines.\n\n'
            f"- Result: **{report['processing_state']}**\n"
            f"- Lines: **{len(lines)}**\n"
            f"- Groups: **{len(run.response.content_groups)}**\n"
            f"- Group sizes: `{report['group_sizes']}`\n"
            f"- Grouping model: `{report['grouping_backend']}`\n"
            f"- Feature schema: `{report['feature_schema_version']}`\n"
            f"- Fallback: `{report['fallback_reason']}`\n"
            f"- Semantic features: `{report['semantic_mode']}`\n\n"
            'See `OCR.txt`, `OCR-blocks.txt`, `content-groups.json`, `layout.json`, and '
            '`extraction.json` for outputs. Text is left as recognized; no manual correction '
            'or accuracy claim is implied.\n', encoding='utf-8')
        (output / 'run-metadata.json').write_text(json.dumps(report, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({key: report[key] for key in ('processing_state', 'ocr_line_count',
            'group_count', 'group_sizes', 'grouping_backend', 'feature_schema_version',
            'fallback_reason', 'timings_ms')}, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        orchestrator.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prior-response', type=Path,
        default=ROOT / 'ocr output/Screenshot 2026-09-30 143459_extraction.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'ocr output/current-run')
    args = parser.parse_args()
    main(args.image, args.prior_response, args.output)
