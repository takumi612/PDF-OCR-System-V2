from __future__ import annotations

import hashlib
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from PIL import Image

from .config import Settings
from .models import PageImage


class PdfValidationError(ValueError):
    def __init__(self, error_code: str, message_vi: str) -> None:
        super().__init__(message_vi)
        self.error_code = error_code
        self.message_vi = message_vi


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class PdfDocumentSession:
    """Giữ PdfReader và PdfDocument mở một lần trong toàn bộ request."""

    def __init__(self, path: Path, settings: Settings) -> None:
        self.path = Path(path).resolve()
        self.settings = settings
        self.reader = None
        self.document = None
        self.page_count = 0
        self.sha256 = ""

    def __enter__(self) -> "PdfDocumentSession":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        if self.path.stat().st_size > self.settings.max_pdf_bytes:
            raise PdfValidationError("pdf_too_large", "PDF vượt giới hạn của pipeline")
        from pypdf import PdfReader
        import pypdfium2 as pdfium

        try:
            self.reader = PdfReader(str(self.path), strict=False)
        except Exception as exc:
            raise PdfValidationError("invalid_pdf", "Không đọc được cấu trúc PDF") from exc
        if self.reader.is_encrypted:
            try:
                if self.reader.decrypt("") == 0:
                    raise PdfValidationError("password_required", "PDF yêu cầu mật khẩu")
            except PdfValidationError:
                raise
            except Exception as exc:
                raise PdfValidationError("password_required", "PDF yêu cầu mật khẩu") from exc
        try:
            self.page_count = len(self.reader.pages)
        except Exception as exc:
            raise PdfValidationError("invalid_pdf", "Không đọc được danh sách trang PDF") from exc
        if self.page_count <= 0:
            raise PdfValidationError("invalid_pdf", "PDF không có trang")
        if self.page_count > self.settings.max_pages:
            raise PdfValidationError("too_many_pages", "PDF vượt giới hạn số trang")
        try:
            self.document = pdfium.PdfDocument(str(self.path))
        except Exception as exc:
            raise PdfValidationError("invalid_pdf", "PDFium không mở được PDF") from exc
        self.sha256 = sha256_file(self.path)

    def close(self) -> None:
        if self.document is not None:
            try:
                self.document.close()
            except Exception:
                pass
            self.document = None
        self.reader = None

    def rotation(self, page_index: int) -> int:
        assert self.reader is not None
        return int(self.reader.pages[page_index].get("/Rotate", 0) or 0) % 360

    def extract_native_text(self, page_index: int) -> str:
        """Đọc text layer bằng PDFium mà không render trang hoặc nạp model OCR."""
        if self.document is None:
            raise RuntimeError("PDF session chưa được mở")
        page = self.document[page_index]
        text_page = None
        try:
            text_page = page.get_textpage()
            return str(text_page.get_text_range() or "")
        finally:
            if text_page is not None:
                try:
                    text_page.close()
                except Exception:
                    pass
            page.close()

    def page_has_content_stream(self, page_index: int) -> bool:
        """Phân biệt trang trắng với trang có ảnh/vector nhưng không có text layer."""
        assert self.reader is not None
        try:
            contents = self.reader.pages[page_index].get_contents()
            if contents is None:
                return False
            data = contents.get_data()
            return bool(data and data.strip())
        except Exception:
            # Không xác định được thì chọn hướng an toàn: coi là có nội dung để OCR.
            return True

    def render_page(self, page_index: int) -> PageImage:
        if self.document is None:
            raise RuntimeError("PDF session chưa được mở")
        page = self.document[page_index]
        try:
            scale = self.settings.fallback_render_dpi / 72.0
            image: Image.Image = page.render(scale=scale).to_pil().convert("RGB")
        finally:
            page.close()
        return PageImage(
            page_index=page_index,
            image=image,
            rotation=self.rotation(page_index),
        )
