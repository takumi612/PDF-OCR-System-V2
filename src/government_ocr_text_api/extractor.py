from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .logging_utils import log_event
from .memory import TensorCleanupController
from .models import ExtractResponse, PageResult
from .native import PdfInspectorNativeExtractor
from .ocr_pipeline import OcrPagePipeline
from .pdf_session import PdfDocumentSession


def markdown_to_text(markdown: str) -> str:
    text = re.sub(r"<!--\s*Page\s+\d+\s*-->", "", markdown, flags=re.IGNORECASE)
    text = re.sub(r"```[^\n]*\n", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~`]", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    # Markdown table: bỏ hàng separator, giữ cell bằng tab.
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\|?[\s:|-]+\|?", stripped) and "-" in stripped:
            continue
        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            line = "\t".join(cells)
        rows.append(line)
    text = "\n".join(rows)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class HybridPdfTextExtractor:
    def __init__(
        self,
        settings: Settings,
        native: PdfInspectorNativeExtractor,
        ocr_pipeline: OcrPagePipeline,
        logger: Any,
    ) -> None:
        self.settings = settings
        self.native = native
        self.ocr_pipeline = ocr_pipeline
        self.logger = logger

    def extract(self, path: Path, filename: str) -> ExtractResponse:
        started = time.perf_counter()
        warnings: list[str] = []
        pages: list[PageResult] = []
        cleanup = TensorCleanupController(self.settings)

        with PdfDocumentSession(path, self.settings) as session:
            log_event(
                self.logger,
                "document_started",
                "Bắt đầu bóc tách PDF",
                filename=filename,
                page_count=session.page_count,
                sha256=session.sha256,
            )
            native_doc = self.native.extract(session)
            if native_doc.fallback_reason:
                warnings.append(native_doc.fallback_reason)

            routed_ocr_pages = [
                page.page_index + 1 for page in native_doc.pages if page.needs_ocr
            ]
            route_reasons = {
                str(page.page_index + 1): page.ocr_reason
                for page in native_doc.pages
                if page.needs_ocr and page.ocr_reason
            }
            log_event(
                self.logger,
                "native_routing_complete",
                "Đã phân tuyến native/OCR",
                native_processing_ms=native_doc.native_processing_ms,
                native_page_count=sum(not page.needs_ocr for page in native_doc.pages),
                ocr_page_count=len(routed_ocr_pages),
                pages_needing_ocr=routed_ocr_pages,
                ocr_reasons=route_reasons,
                fallback_reason=native_doc.fallback_reason,
            )

            pending_ocr: list[tuple[int, str | None]] = []

            def append_and_log(page_result: PageResult) -> None:
                pages.append(page_result)
                cleanup.after_page(len(pages))
                log_event(
                    self.logger,
                    "page_complete",
                    "Đã xử lý xong trang",
                    page_index=page_result.page_index,
                    source=page_result.source,
                    page_count=session.page_count,
                    line_count=page_result.line_count,
                    needs_review=page_result.needs_review,
                    metrics=page_result.metrics,
                )

            def flush_ocr_window() -> None:
                if not pending_ocr:
                    return
                if hasattr(self.ocr_pipeline, "extract_pages"):
                    results = self.ocr_pipeline.extract_pages(session, list(pending_ocr))
                else:  # backward-compatible test/custom adapter
                    results = [
                        self.ocr_pipeline.extract_page(session, page_index, reason)
                        for page_index, reason in pending_ocr
                    ]
                pending_ocr.clear()
                for result in results:
                    append_and_log(result)

            for native_page in native_doc.pages:
                if native_page.needs_ocr:
                    pending_ocr.append((native_page.page_index, native_page.ocr_reason))
                    if len(pending_ocr) >= self.settings.recognition_window_pages:
                        flush_ocr_window()
                    continue

                # Flush trước trang native để log và kết quả vẫn theo thứ tự tài liệu.
                flush_ocr_window()
                text = markdown_to_text(native_page.markdown)
                append_and_log(
                    PageResult(
                        page_index=native_page.page_index,
                        page_number=native_page.page_index + 1,
                        source="native",
                        text=text,
                        markdown=native_page.markdown,
                        needs_ocr=False,
                        ocr_reason=None,
                        line_count=len([line for line in text.splitlines() if line.strip()]),
                        metrics={"route": "native"},
                        ai_safe_text=text,
                    )
                )

            flush_ocr_window()

            cleanup.document_end()
            sha256 = session.sha256

        pages.sort(key=lambda page: page.page_index)
        text_parts: list[str] = []
        ai_safe_text_parts: list[str] = []
        markdown_parts: list[str] = []
        for page in pages:
            page_ai_safe_text = page.ai_safe_text
            if (
                not page_ai_safe_text
                and page.ai_ready
                and page.semantic_risk_count == 0
            ):
                # Backward compatibility for native/custom adapters that predate
                # the explicit AI-safe channel.
                page_ai_safe_text = page.text
            if self.settings.include_page_markers:
                text_parts.append(f"===== Trang {page.page_number} =====\n{page.text}".rstrip())
                ai_safe_text_parts.append(
                    f"===== Trang {page.page_number} =====\n{page_ai_safe_text}".rstrip()
                )
                markdown_parts.append(
                    f"<!-- Page {page.page_number} -->\n\n{page.markdown or page.text}".rstrip()
                )
            else:
                text_parts.append(page.text)
                ai_safe_text_parts.append(page_ai_safe_text)
                markdown_parts.append(page.markdown or page.text)

        semantic_risk_count = sum(page.semantic_risk_count for page in pages)
        ai_ready = all(page.ai_ready for page in pages) and semantic_risk_count == 0
        if semantic_risk_count and "semantic_review_required" not in warnings:
            warnings.append("semantic_review_required")
        status = (
            "partial"
            if any(page.error_codes for page in pages) or not ai_ready
            else "complete"
        )
        total_ms = round((time.perf_counter() - started) * 1000, 3)
        metrics = {
            "native_processing_ms": native_doc.native_processing_ms,
            "cleanup_count": cleanup.cleanup_count,
            "pages_with_tables": native_doc.pages_with_tables,
            "pages_with_columns": native_doc.pages_with_columns,
            "pages_needing_ocr": [
                page.page_number for page in pages if page.source == "ocr"
            ],
            "ocr_reasons": {
                str(page.page_number): page.ocr_reason
                for page in pages
                if page.source == "ocr" and page.ocr_reason
            },
            "native_fallback_reason": native_doc.fallback_reason,
            "semantic_risk_count": semantic_risk_count,
            "ai_ready": ai_ready,
        }
        log_event(
            self.logger,
            "document_complete",
            "Đã hoàn tất bóc tách PDF",
            filename=filename,
            status=status,
            processing_time_ms=total_ms,
            native_page_count=sum(page.source == "native" for page in pages),
            ocr_page_count=sum(page.source == "ocr" for page in pages),
            semantic_risk_count=semantic_risk_count,
            ai_ready=ai_ready,
        )
        return ExtractResponse(
            filename=filename,
            sha256=sha256,
            page_count=len(pages),
            native_page_count=sum(page.source == "native" for page in pages),
            ocr_page_count=sum(page.source == "ocr" for page in pages),
            status=status,
            processing_time_ms=total_ms,
            text="\n\n".join(text_parts).strip(),
            ai_safe_text="\n\n".join(ai_safe_text_parts).strip(),
            ai_ready=ai_ready,
            semantic_risk_count=semantic_risk_count,
            markdown="\n\n".join(markdown_parts).strip()
            if self.settings.include_markdown_in_response
            else None,
            pages=pages if self.settings.include_page_results else None,
            metrics=metrics,
            warnings=warnings,
        )
