from pathlib import Path

from fastapi.testclient import TestClient

from government_ocr_text_api.main import app, engine, settings
from government_ocr_text_api.models import ExtractResponse


def test_extract_endpoint_accepts_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "request_temp_root", tmp_path)
    def fake_extract(path: Path, filename: str):
        assert path.read_bytes().startswith(b"%PDF-")
        return ExtractResponse(
            filename=filename,
            sha256="abc",
            page_count=1,
            native_page_count=1,
            ocr_page_count=0,
            status="complete",
            processing_time_ms=1.0,
            text="Nội dung",
            markdown="Nội dung",
            pages=[],
        )

    monkeypatch.setattr(engine, "extract", fake_extract)
    with TestClient(app) as client:
        response = client.post(
            "/api/extract",
            files={"file": ("mau.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json()["text"] == "Nội dung"


def test_extract_endpoint_rejects_non_pdf():
    with TestClient(app) as client:
        response = client.post(
            "/api/extract",
            files={"file": ("mau.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 400
