import asyncio
from types import SimpleNamespace
import threading
import time

import pytest

from app.core.config import Settings
from app.modules.ocr.adapter import lines_to_text_blocks
from app.modules.ocr.service import OCRService
from app.modules.ocr.types import OCRLine
from app.orchestration.extractor import ExtractionOrchestrator
from app.schemas.common import BBox
from tests.unit.test_model_adapters import _prepared_page


def test_line_adapter_ids_and_normalization_are_stable_with_duplicates():
    page = _prepared_page()
    lines = [OCRLine('  كتاب  ', .9, BBox(x1=100,y1=100,x2=500,y2=130))]*2
    first = lines_to_text_blocks(lines,page=page)
    second = lines_to_text_blocks(list(reversed(lines)),page=page)
    assert [b.block_id for b in first] == [b.block_id for b in second]
    assert len({b.block_id for b in first}) == 2
    assert all(b.original_text == '  كتاب  ' and b.normalized_text == 'کتاب' for b in first)


def test_line_adapter_preserves_available_layout_metadata():
    line = OCRLine('Section',.9,BBox(x1=100,y1=100,x2=500,y2=130),column_id='left',block_type='heading')
    block = lines_to_text_blocks([line],page=_prepared_page())[0]
    assert block.column_id == 'left' and block.block_type == 'heading'


@pytest.mark.asyncio
async def test_live_grouping_can_preserve_separation_inside_heuristic_paragraph():
    page = _prepared_page()
    settings = Settings(_env_file=None,grouping_enabled=True,grouping_backend='clustering',ocr_backend='mock')
    service = OCRService(settings)
    settings.ocr_backend = 'paddle'
    service._paddle_backend = SimpleNamespace(predict=lambda image:[
        OCRLine('First complete paragraph.',.9,BBox(x1=100,y1=100,x2=500,y2=130)),
        OCRLine('Another complete paragraph.',.9,BBox(x1=100,y1=150,x2=500,y2=180))])
    orchestrator = ExtractionOrchestrator(settings,ocr=service)
    runs = [await orchestrator.extract_document_run(document_id=page.document_id,pages=[page],request_id=str(i),document_metadata={}) for i in range(2)]
    assert len(runs[0].pages[0].ocr_analysis.heuristic_objects) == 1
    assert len(runs[0].response.content_groups) == 2
    assert [g.group_id for g in runs[0].response.content_groups] == [g.group_id for g in runs[1].response.content_groups]
    assert 'ocr_analysis' not in runs[0].response.model_dump()
    orchestrator.close()


@pytest.mark.asyncio
async def test_concurrent_grouping_is_off_loop_and_document_local(monkeypatch):
    settings = Settings(_env_file=None,grouping_enabled=True,grouping_backend='clustering')
    orchestrator = ExtractionOrchestrator(settings)
    main_thread = threading.get_ident()
    threads = []
    original = orchestrator._apply_learned_grouping
    def delayed(*args):
        threads.append(threading.get_ident())
        time.sleep(.08)
        return original(*args)
    monkeypatch.setattr(orchestrator,'_apply_learned_grouping',delayed)
    page = _prepared_page()
    ticks = 0
    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(.01)
            ticks += 1
    async def extract(i):
        return await orchestrator.extract_document(document_id=page.document_id,pages=[page],request_id=str(i),document_metadata={})
    first,second,_ = await asyncio.gather(extract(1),extract(2),heartbeat())
    assert ticks == 10 and all(t != main_thread for t in threads)
    assert [g.group_id for g in first.content_groups] == [g.group_id for g in second.content_groups]
    orchestrator.close()
