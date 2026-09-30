from app.artifacts.publisher import ArtifactPublisher
from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox
from app.schemas.detection import DetectedObject, ObjectType, Provenance
from app.schemas.status import ModuleName


def _line(text: str, y: float) -> OCRLine:
    return OCRLine(
        text=text,
        confidence=0.95,
        bbox=BBox(x1=100, y1=y, x2=900, y2=y + 30),
    )


def test_legal_list_items_become_separate_readable_paragraphs():
    lines = [
        _line("ماده ٧ وظايف رئيس كميسيون به شرح زير است", 100),
        _line("الف اداره جلسات كميسيون", 140),
        _line("ب ابلاغ مصوبات پس از تأييد رئيس جمهور", 180),
        _line("ج صدور احكام رؤسای شورای معين", 220),
        _line("د ارسال گزارش عملكرد", 260),
        _line("ه انتصاب دبير كميسيون", 300),
    ]

    paragraphs = group_lines_into_paragraphs(lines)

    assert [paragraph.text for paragraph in paragraphs] == [
        "ماده ٧ وظايف رئيس كميسيون به شرح زير است",
        "الف - اداره جلسات كميسيون",
        "ب - ابلاغ مصوبات پس از تأييد رئيس جمهور",
        "ج - صدور احكام رؤسای شورای معين",
        "د - ارسال گزارش عملكرد",
        "ه - انتصاب دبير كميسيون",
    ]
    assert paragraphs[1].raw_text == "الف اداره جلسات كميسيون"


def test_glued_persian_marker_is_recovered_only_inside_sequence():
    lines = [
        _line("ج پیگیری تشکیل شورای معین", 100),
        _line("داطلاع رسانی برنامهها و تصميمات", 140),
        _line("ه شركت در جلسات مرتبط", 180),
        _line("و هماهنگی با دستگاههای اجرایی", 220),
        _line("ز ثبت و پیگیری شکایات", 260),
        _line("حتدوين گزارش ساليانه", 300),
    ]

    paragraphs = group_lines_into_paragraphs(lines)

    assert [paragraph.text for paragraph in paragraphs] == [
        "ج - پیگیری تشکیل شورای معین",
        "د - اطلاع رسانی برنامهها و تصميمات",
        "ه - شركت در جلسات مرتبط",
        "و - هماهنگی با دستگاههای اجرایی",
        "ز - ثبت و پیگیری شکایات",
        "ح - تدوين گزارش ساليانه",
    ]


def test_ordinary_wrapped_line_starting_with_vav_is_not_a_list_item():
    lines = [
        _line("ماده ٢ جلسات كميسيون هر دو ماه يك بار تشكيل مي شود", 100),
        _line("و در صورت لزوم با نظر رئيس كميسيون يا سه عضو از اعضا", 140),
        _line("خواهد شد دستور جلسه حداقل يك هفته قبل از برگزاري جلسات", 180),
    ]

    paragraphs = group_lines_into_paragraphs(lines)

    assert len(paragraphs) == 1
    assert paragraphs[0].text == (
        "ماده ٢ جلسات كميسيون هر دو ماه يك بار تشكيل مي شود\n"
        "و در صورت لزوم با نظر رئيس كميسيون يا سه عضو از اعضا\n"
        "خواهد شد دستور جلسه حداقل يك هفته قبل از برگزاري جلسات"
    )


def test_plain_text_prefers_normalized_text_but_keeps_raw_text_in_object():
    obj = DetectedObject(
        object_id="paragraph-1",
        document_id="doc-1",
        page_id="doc-1:p1",
        page_number=1,
        type=ObjectType.PARAGRAPH,
        bbox=BBox(x1=0, y1=0, x2=100, y2=20),
        confidence=0.95,
        text="الف - اداره جلسات كميسيون",
        raw_text="الف اداره جلسات كميسيون",
        provenance=Provenance(
            module=ModuleName.OCR,
            backend="paddle",
            model_id="test",
        ),
    )

    assert ArtifactPublisher._ocr_plain_text([obj]) == "الف - اداره جلسات كميسيون\n"
    assert obj.raw_text == "الف اداره جلسات كميسيون"
