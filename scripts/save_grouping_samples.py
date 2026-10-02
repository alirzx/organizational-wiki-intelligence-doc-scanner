"""Save synthetic inputs and actual postprocessing/publisher outputs locally."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Optional dependency installed for the implementation's native smoke run.
LOCAL_PACKAGES = ROOT / '.artifacts/postprocessing-implementation/python-packages'
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import httpx
from PIL import Image

from app.artifacts.publisher import ArtifactPublisher
from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_detected_objects, lines_to_text_blocks
from app.modules.ocr.types import OCRAnalysis, OCRLine
from app.orchestration.extractor import ExtractionOrchestrator, PageRunResult
from app.preprocessing.types import PreparedPage
from app.schemas.common import BBox
from app.schemas.extraction import DocumentExtractionResponse, ModulePageResponse, PageExtractionResponse
from app.schemas.image import ImageMetadata, TransformMetadata
from app.schemas.status import ModuleName, ModuleStatus, ProcessingStatus
from app.text_processing.embeddings.cache import RunEmbeddingCache
from app.text_processing.embeddings.ollama_embedding import OllamaEmbedder


def jsonable(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if hasattr(value, '__dataclass_fields__'):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


class LocalSampleStorage:
    """Publisher storage sink; no network access or cleanup of existing files."""
    def __init__(self, directory, document_id):
        self.directory = directory.resolve()
        self.prefix = ArtifactPublisher._document_prefix(document_id) + '/'

    def delete_prefix(self, prefix):
        # Samples overwrite known files only; never delete user files.
        pass

    def target(self, key):
        if not key.startswith(self.prefix):
            raise ValueError('unexpected artifact document prefix')
        target = (self.directory / key[len(self.prefix):]).resolve()
        if not target.is_relative_to(self.directory):
            raise ValueError('artifact escaped sample directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def put_text(self, key, text, **kwargs):
        self.target(key).write_text(text, encoding='utf-8')

    def put_bytes(self, key, data, **kwargs):
        self.target(key).write_bytes(data)


class RetainedOCRSampleOrchestrator(ExtractionOrchestrator):
    """Replace OCR inference only; use real adapters and document grouping."""
    async def _run_page(self, page, request_id, page_semaphore):
        lines = self.sample_lines[page.page_number]
        objects = lines_to_detected_objects(lines, page=page, settings=self.settings,
                                            backend_name='synthetic-ocr-input')
        blocks = lines_to_text_blocks(lines, page=page)
        status = ModuleStatus(module=ModuleName.OCR, state='success', duration_ms=0,
                              backend='synthetic-ocr-input', model_id='no-ocr-model')
        module = ModulePageResponse(schema_version=self.settings.schema_version,
            request_id=request_id, document_id=page.document_id, page_id=page.page_id,
            page_number=page.page_number, module=ModuleName.OCR, image=page.image_metadata,
            transform=page.transform, objects=objects, status=status)
        response = PageExtractionResponse(schema_version=self.settings.schema_version,
            request_id=request_id, document_id=page.document_id, page_id=page.page_id,
            page_number=page.page_number, image=page.image_metadata, transform=page.transform,
            objects=objects, modules={ModuleName.OCR: status},
            processing=ProcessingStatus(state='success', duration_ms=0))
        self.canonical.extend(blocks)
        return PageRunResult(page, {ModuleName.OCR: module}, response,
                             OCRAnalysis(tuple(lines), tuple(blocks), tuple(objects)))


def line(text, y, *, x=100, width=600, column='main', kind='body'):
    return OCRLine(text=text, confidence=.98, bbox=BBox(x1=x, y1=y, x2=x+width, y2=y+20),
                   column_id=column, block_type=kind)


def fixtures():
    english = {1: [line('The scanner keeps each original OCR line', 100),
                   line('and joins nearby lines into readable paragraphs.', 126),
                   line('A separate paragraph starts after a larger gap.', 240)]}
    return {
        'english': english,
        'persian-list': {1: [line('راهنمای پردازش اسناد', 80, kind='heading'),
                            line('اين متن با حروف عربي ي و ك نوشته شده است', 120),
                            line('و ادامه همان پاراگراف فارسی است.', 146),
                            line('۱. بررسی تصویر', 260),
                            line('و ثبت نتیجه بررسی', 286),
                            line('۲. ذخیره خروجی', 330)]},
        'columns': {1: [line('Left column paragraph begins', 100, width=300, column='left'),
                       line('Right column paragraph begins', 100, x=600, width=300, column='right'),
                       line('and continues on the left.', 126, width=300, column='left'),
                       line('and continues on the right.', 126, x=600, width=300, column='right')]},
        'cross-page': {1: [line('This paragraph starts near the end of the page', 1320),
                           line('and continues onto the next page', 1346)],
                       2: [line('and finishes in the first line of page two.', 20),
                           line('A later paragraph is independent.', 180)]},
    }


def mock_embedder(mode):
    def handler(request):
        if mode == 'failure':
            return httpx.Response(503, json={'error': 'synthetic provider unavailable'})
        texts = json.loads(request.content)['input']
        return httpx.Response(200, json={'embeddings': [[1., 0., 0., 0.] for _ in texts]})
    return OllamaEmbedder(base_url='http://synthetic-provider.invalid', dimensions=4,
        model='synthetic-unit-vectors', max_attempts=1, retry_backoff=0,
        cache=RunEmbeddingCache(100), transport=httpx.MockTransport(handler))


def generate(output, model_package=None):
    scenes = fixtures()
    cases = [(name, name, 'clustering', 'off') for name in scenes]
    cases += [('semantic-mock-success', 'english', 'clustering', 'success'),
              ('semantic-mock-failure', 'english', 'clustering', 'failure')]
    if model_package is not None:
        cases.append(('lightgbm-synthetic-model', 'english', 'lightgbm', 'off'))
    summary = []
    for name, scene, backend, semantics in cases:
        directory = output / name
        document_id = 'sample-' + name
        settings = Settings(_env_file=None, grouping_enabled=True, grouping_backend=backend,
            grouping_model_path=str(model_package) if model_package else 'unused',
            semantic_features_enabled=semantics != 'off', semantic_failure_policy='fallback')
        orchestrator = RetainedOCRSampleOrchestrator(settings,
            grouping_embedder=mock_embedder(semantics) if semantics != 'off' else None)
        orchestrator.sample_lines = scenes[scene]
        orchestrator.canonical = []
        pages = []
        for number in scenes[scene]:
            image = Image.new('RGB', (1000, 1400), 'white')
            pages.append(PreparedPage(document_id, f'{document_id}:p{number}', number, {},
                image, image, ImageMetadata(filename=f'synthetic-page-{number}.png',
                    mime_type='image/png', source_width=1000, source_height=1400,
                    processed_width=1000, processed_height=1400),
                TransformMetadata(scale_x=1, scale_y=1)))
        try:
            run = asyncio.run(orchestrator.extract_document_run(document_id=document_id,
                pages=pages, request_id='sample-generation',
                document_metadata={'synthetic': True, 'ocr_inference': False, 'scenario': scene}))
            save_json(directory / 'input.json', {'synthetic': True, 'document_id': document_id,
                'backend': backend, 'semantics': semantics, 'page_width': 1000, 'page_height': 1400,
                'pages': [{'page_number': n, 'lines': jsonable(lines)} for n, lines in scenes[scene].items()]})
            save_json(directory / 'canonical-blocks.json', orchestrator.canonical)
            save_json(directory / 'extraction.json', run.response)
            save_json(directory / 'diagnostics.json', run.response.grouping)
            keys = ArtifactPublisher(LocalSampleStorage(directory, document_id)).publish(run)
            # Validate public response round-trip and lossless block membership.
            DocumentExtractionResponse.model_validate_json((directory / 'extraction.json').read_text(encoding='utf-8'))
            members = [m.block_id for g in run.response.content_groups for m in g.members]
            assert sorted(members) == sorted(b.block_id for b in orchestrator.canonical)
            expected_semantics = {'off': 'disabled', 'success': 'available', 'failure': 'unavailable'}[semantics]
            assert run.response.grouping.semantic_mode == expected_semantics
            for path in directory.rglob('*'):
                if path.is_file() and (path.suffix == '.json' or path.name.endswith('.json.txt')):
                    json.loads(path.read_text(encoding='utf-8'))
            summary.append({'sample': name, 'backend': backend, 'semantic_mode': expected_semantics,
                'blocks': len(orchestrator.canonical), 'groups': len(run.response.content_groups),
                'pages': len(pages), 'publisher_artifacts': len(keys)})
        finally:
            orchestrator.close()
    save_json(output / 'manifest.json', {'synthetic': True, 'samples': summary,
        'validation': 'Public responses round-trip; all input blocks appear exactly once; JSON parses; semantic modes match.'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'samples/postprocessing')
    parser.add_argument('--model-package', type=Path,
                        help='Optional native package; use the synthetic smoke package for sample provenance.')
    args = parser.parse_args()
    generate(args.output, args.model_package)
