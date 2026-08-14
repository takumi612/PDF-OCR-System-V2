import sys
from pathlib import Path
from types import SimpleNamespace

from government_ocr_text_api.config import NativeFailurePolicy, Settings
from government_ocr_text_api.native import PdfInspectorNativeExtractor


class FakeSession:
    def __init__(self, page_count=2):
        self.path = Path("x.pdf")
        self.page_count = page_count
        self.text_by_page = {}
        self.content_by_page = {}

    def extract_native_text(self, page_index):
        return self.text_by_page.get(page_index, "")

    def page_has_content_stream(self, page_index):
        return self.content_by_page.get(page_index, False)


def test_native_maps_page_flags(monkeypatch):
    fake = SimpleNamespace(
        extract_pages_markdown=lambda path: SimpleNamespace(
            pages=[
                SimpleNamespace(page=0, markdown="Nội dung", needs_ocr=False, ocr_reason=None),
                SimpleNamespace(page=1, markdown="", needs_ocr=True, ocr_reason="scanned"),
            ],
            pages_with_tables=[1],
            pages_with_columns=[],
        )
    )
    monkeypatch.setitem(sys.modules, "pdf_inspector", fake)
    result = PdfInspectorNativeExtractor(Settings()).extract(FakeSession())
    assert result.pages[0].needs_ocr is False
    assert result.pages[1].needs_ocr is True
    assert result.pages[1].ocr_reason == "scanned"


def test_short_or_blank_native_page_does_not_force_ocr(monkeypatch):
    fake = SimpleNamespace(
        extract_pages_markdown=lambda path: SimpleNamespace(
            pages=[SimpleNamespace(page=0, markdown="1", needs_ocr=False, ocr_reason=None)],
            pages_with_tables=[],
            pages_with_columns=[],
        )
    )
    monkeypatch.setitem(sys.modules, "pdf_inspector", fake)
    result = PdfInspectorNativeExtractor(Settings(native_min_text_chars=5)).extract(
        FakeSession(page_count=1)
    )
    assert result.pages[0].needs_ocr is False
    assert result.pages[0].markdown == "1"


def test_pdf_inspector_failure_uses_pdfium_before_ocr(monkeypatch):
    fake = SimpleNamespace(
        extract_pages_markdown=lambda path: (_ for _ in ()).throw(RuntimeError("broken"))
    )
    monkeypatch.setitem(sys.modules, "pdf_inspector", fake)
    session = FakeSession(page_count=2)
    session.text_by_page = {0: "Text layer đầy đủ", 1: ""}
    session.content_by_page = {0: True, 1: False}

    result = PdfInspectorNativeExtractor(
        Settings(native_failure_policy=NativeFailurePolicy.PDFIUM_THEN_OCR)
    ).extract(session)

    assert result.pages[0].needs_ocr is False
    assert result.pages[1].needs_ocr is False  # blank page
    assert result.fallback_reason.startswith("native_failure:RuntimeError")


def test_short_signature_text_does_not_override_explicit_scan_flag(monkeypatch):
    fake = SimpleNamespace(
        extract_pages_markdown=lambda path: SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page=0,
                    markdown="Đã ký số",
                    needs_ocr=True,
                    ocr_reason="full_page_image",
                )
            ],
            pages_with_tables=[],
            pages_with_columns=[],
        )
    )
    monkeypatch.setitem(sys.modules, "pdf_inspector", fake)
    result = PdfInspectorNativeExtractor(Settings(native_min_text_chars=5)).extract(
        FakeSession(page_count=1)
    )
    assert result.pages[0].needs_ocr is True
    assert result.pages[0].ocr_reason == "full_page_image"
