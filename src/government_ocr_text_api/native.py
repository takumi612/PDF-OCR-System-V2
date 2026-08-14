from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from .config import NativeFailurePolicy, Settings
from .models import NativeDocument, NativePage

if TYPE_CHECKING:
    from .pdf_session import PdfDocumentSession


def _reason_value(page: Any) -> str | None:
    value = getattr(page, "ocr_reason", None)
    if value:
        return str(value)
    reasons = getattr(page, "ocr_reasons", None)
    if isinstance(reasons, (list, tuple)) and reasons:
        return ",".join(str(reason) for reason in reasons)
    return None


class PdfInspectorNativeExtractor:
    """Trích xuất text native và quyết định OCR theo từng trang.

    Nguyên tắc quan trọng:
    - Tin cờ ``needs_ocr`` do pdf-inspector trả về.
    - Không ép trang native ngắn hoặc trang trắng sang OCR chỉ vì ít ký tự.
    - Khi pdf-inspector lỗi, thử text layer bằng PDFium trước khi OCR.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(self, session: "PdfDocumentSession") -> NativeDocument:
        started = time.perf_counter()
        page_count = session.page_count
        if not self.settings.native_pdf_enabled:
            return self._ocr_all(page_count, "native_pdf_disabled", started)

        try:
            import pdf_inspector

            result = pdf_inspector.extract_pages_markdown(str(session.path))
            raw_pages = sorted(result.pages, key=lambda item: int(item.page))
            pages_by_index = {int(item.page): item for item in raw_pages}
            pages: list[NativePage] = []

            for page_index in range(page_count):
                item = pages_by_index.get(page_index)
                if item is None:
                    pages.append(self._fallback_page(session, page_index, "native_page_missing"))
                    continue

                markdown = str(getattr(item, "markdown", "") or "")
                reported_needs_ocr = bool(getattr(item, "needs_ocr", False))
                reason = _reason_value(item)

                # Bug cũ: markdown dưới 5 ký tự luôn bị OCR, kể cả trang trắng hoặc
                # trang chỉ có số trang. Điều đó làm nạp Paddle/VietOCR không cần thiết.
                if reported_needs_ocr:
                    pages.append(
                        NativePage(
                            page_index=page_index,
                            markdown=markdown,
                            needs_ocr=True,
                            ocr_reason=reason or "pdf_inspector_requires_ocr",
                        )
                    )
                    continue

                if (
                    not self.settings.native_keep_short_text_pages
                    and 0 < len(markdown.strip()) < self.settings.native_min_text_chars
                ):
                    pages.append(self._fallback_page(session, page_index, "native_text_too_short"))
                    continue

                pages.append(
                    NativePage(
                        page_index=page_index,
                        markdown=markdown,
                        needs_ocr=False,
                        ocr_reason=None,
                    )
                )

            return NativeDocument(
                pages=pages,
                pages_with_tables=list(getattr(result, "pages_with_tables", []) or []),
                pages_with_columns=list(getattr(result, "pages_with_columns", []) or []),
                native_processing_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        except Exception as exc:
            reason = f"native_failure:{type(exc).__name__}:{exc}"
            if self.settings.native_failure_policy is NativeFailurePolicy.FAIL:
                raise RuntimeError(f"pdf-inspector không xử lý được PDF: {exc}") from exc
            if self.settings.native_failure_policy is NativeFailurePolicy.PDFIUM_THEN_OCR:
                return self._pdfium_fallback(session, reason, started)
            return self._ocr_all(page_count, reason, started)

    def _pdfium_fallback(
        self,
        session: "PdfDocumentSession",
        reason: str,
        started: float,
    ) -> NativeDocument:
        pages = [
            self._fallback_page(session, page_index, reason)
            for page_index in range(session.page_count)
        ]
        return NativeDocument(
            pages=pages,
            native_processing_ms=round((time.perf_counter() - started) * 1000, 3),
            fallback_reason=reason,
        )

    def _fallback_page(
        self,
        session: "PdfDocumentSession",
        page_index: int,
        reason: str,
    ) -> NativePage:
        try:
            text = session.extract_native_text(page_index)
        except Exception as exc:
            return NativePage(
                page_index=page_index,
                markdown="",
                needs_ocr=True,
                ocr_reason=f"{reason};pdfium_text_failure:{type(exc).__name__}",
            )

        stripped = text.strip()
        if len(stripped) >= self.settings.native_min_text_chars:
            return NativePage(
                page_index=page_index,
                markdown=text,
                needs_ocr=False,
                ocr_reason=None,
            )

        # Trang thật sự trắng không cần nạp OCR. Trang có content stream nhưng
        # không đọc được text có thể là scan, vector text hoặc encoding lỗi.
        if not session.page_has_content_stream(page_index):
            return NativePage(
                page_index=page_index,
                markdown="",
                needs_ocr=False,
                ocr_reason=None,
            )

        return NativePage(
            page_index=page_index,
            markdown=text,
            needs_ocr=True,
            ocr_reason=f"{reason};pdfium_text_missing",
        )

    @staticmethod
    def _ocr_all(page_count: int, reason: str, started: float) -> NativeDocument:
        return NativeDocument(
            pages=[NativePage(index, "", True, reason) for index in range(page_count)],
            native_processing_ms=round((time.perf_counter() - started) * 1000, 3),
            fallback_reason=reason,
        )
