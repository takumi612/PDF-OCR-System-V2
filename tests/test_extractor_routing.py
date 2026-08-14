from pathlib import Path
from types import SimpleNamespace

from government_ocr_text_api.config import Settings
from government_ocr_text_api.extractor import HybridPdfTextExtractor
from government_ocr_text_api.models import NativeDocument, NativePage, PageResult


class FakeSession:
    page_count = 2
    sha256 = "abc"
    def __init__(self, path, settings): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Native:
    def extract(self, session):
        return NativeDocument([
            NativePage(0, "# Native", False),
            NativePage(1, "", True, "scanned"),
        ])


class Ocr:
    def extract_page(self, session, page_index, reason):
        return PageResult(
            page_index=page_index,
            page_number=page_index + 1,
            source="ocr",
            text="OCR",
            markdown="OCR",
            needs_ocr=True,
            ocr_reason=reason,
        )


class Logger:
    def info(self, value): pass


def test_hybrid_routes_only_scan_page(monkeypatch, tmp_path):
    monkeypatch.setattr("government_ocr_text_api.extractor.PdfDocumentSession", FakeSession)
    extractor = HybridPdfTextExtractor(Settings(), Native(), Ocr(), Logger())
    result = extractor.extract(tmp_path / "x.pdf", "x.pdf")
    assert result.native_page_count == 1
    assert result.ocr_page_count == 1
    assert "Native" in result.text
    assert "OCR" in result.text


class WindowSession:
    page_count = 5
    sha256 = "window"
    def __init__(self, path, settings): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class AllOcrNative:
    def extract(self, session):
        return NativeDocument([NativePage(index, "", True, "scanned") for index in range(5)])


class WindowOcr:
    def __init__(self):
        self.windows = []

    def extract_pages(self, session, page_specs):
        self.windows.append([index for index, _ in page_specs])
        return [
            PageResult(
                page_index=index,
                page_number=index + 1,
                source="ocr",
                text=f"OCR {index}",
                markdown=f"OCR {index}",
                needs_ocr=True,
                ocr_reason=reason,
            )
            for index, reason in page_specs
        ]


def test_ocr_pages_are_processed_in_document_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("government_ocr_text_api.extractor.PdfDocumentSession", WindowSession)
    ocr = WindowOcr()
    extractor = HybridPdfTextExtractor(
        Settings(recognition_window_pages=3),
        AllOcrNative(),
        ocr,
        Logger(),
    )

    result = extractor.extract(tmp_path / "x.pdf", "x.pdf")

    assert ocr.windows == [[0, 1, 2], [3, 4]]
    assert result.ocr_page_count == 5


class SemanticSession:
    page_count = 1
    sha256 = "semantic"

    def __init__(self, path, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class SemanticNative:
    def extract(self, session):
        return NativeDocument([NativePage(0, "", True, "scanned")])


class SemanticOcr:
    def extract_page(self, session, page_index, reason):
        return PageResult(
            page_index=0,
            page_number=1,
            source="ocr",
            text="Nội dung raw chưa chắc chắn",
            markdown="Nội dung raw chưa chắc chắn",
            needs_ocr=True,
            ocr_reason=reason,
            ai_safe_text=(
                "[OCR_SEMANTIC_RISK page=1 line=1 "
                "reasons=secondary_indicates_primary_omission]"
            ),
            ai_ready=False,
            semantic_risk_count=1,
            error_codes=["semantic_risk_detected"],
        )


def test_document_aggregates_ai_safe_channel_and_risk_state(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "government_ocr_text_api.extractor.PdfDocumentSession",
        SemanticSession,
    )
    extractor = HybridPdfTextExtractor(
        Settings(),
        SemanticNative(),
        SemanticOcr(),
        Logger(),
    )

    result = extractor.extract(tmp_path / "x.pdf", "x.pdf")

    assert "Nội dung raw chưa chắc chắn" in result.text
    assert "Nội dung raw chưa chắc chắn" not in result.ai_safe_text
    assert "OCR_SEMANTIC_RISK page=1 line=1" in result.ai_safe_text
    assert result.ai_ready is False
    assert result.semantic_risk_count == 1
    assert "semantic_review_required" in result.warnings
