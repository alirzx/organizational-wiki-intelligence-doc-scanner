from app.modules.ocr.paragraph_grouper import group_lines_into_paragraphs
from app.modules.ocr.types import OCRLine
from app.schemas.common import BBox


def _line(text: str, y: float) -> OCRLine:
    return OCRLine(
        text=text,
        confidence=0.95,
        bbox=BBox(x1=100, y1=y, x2=900, y2=y + 30),
    )


def test_full_persian_alphabetic_list_splits_each_item_and_keeps_wrapped_lines():
    lines = [
        _line("الف-اطالعات:هرنوعدادهازجملهصوت،تصویر،فیلم", 100),
        _line("ذخیرهگردیدهویاباهروسیلهدیگریضبطشدهباشد.", 140),
        _line("ب-اطالعاتشخصی:اطالعاتمربوطبههوّیت", 180),
        _line("عاداترفتاریوفردیازقبیلنامونامخانوادگی", 220),
        _line("پ-حریمخصوصی:قلمروییاززندگیشخصیفرد", 260),
        _line("ت-اطالعاتطبقهبندیشده:اسنادسریومحرمانه", 300),
        _line("ث-قانون:قانونانتشارودسترسیآزادبهاطالعات", 340),
        _line("ج-نشراطالعات:قراردادناطالعاتدرمعرضدسترسیعموم", 380),
        _line("چ-مؤسساتخصوصی:اشخاصحقوقیکهباتجویزقانون", 420),
        _line("ح-مؤسساتعمومی:سازمانهاونهادهایوابستهبهحکومت", 460),
        _line("خ-مؤسساتخصوصیارایهدهندهخدمتعمومی", 500),
        _line("د-کمیسیون:کمیسیونماده(١٨)قانون", 540),
        _line("ذ-مؤسساتمشمولقانون:مؤسساتخصوصیوعمومی", 580),
        _line("ر-درگاه:شاملپرتال،وبسایتووبگاه", 620),
    ]

    paragraphs = group_lines_into_paragraphs(lines)

    assert [p.text.split("\n", 1)[0].split(" - ", 1)[0] for p in paragraphs] == [
        "الف", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ", "د", "ذ", "ر"
    ]
    assert paragraphs[0].text.endswith("\nذخیرهگردیدهویاباهروسیلهدیگریضبطشدهباشد.")
    assert paragraphs[1].text.endswith("\nعاداترفتاریوفردیازقبیلنامونامخانوادگی")


def test_english_alpha_list_splits_each_item_and_keeps_wrapped_lines():
    lines = [
        _line("A. Overview of the service", 100),
        _line("This sentence wraps visually onto the next OCR line.", 140),
        _line("B) Requirements", 180),
        _line("C - Deployment", 220),
    ]

    paragraphs = group_lines_into_paragraphs(lines)

    assert [p.text.split("\n", 1)[0] for p in paragraphs] == [
        "A - Overview of the service",
        "B - Requirements",
        "C - Deployment",
    ]
    assert "\nThis sentence wraps visually onto the next OCR line." in paragraphs[0].text


def test_symbol_and_numeric_bullets_start_new_blocks():
    lines = [
        _line("Intro paragraph", 100),
        _line("• first item", 140),
        _line("- second item", 180),
        _line("1. third item", 220),
        _line("۲) fourth item", 260),
    ]

    paragraphs = group_lines_into_paragraphs(lines)
    assert len(paragraphs) == 5
