from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .engine import ExtractionEngine
from .models import ErrorResponse, ExtractResponse
from .pdf_session import PdfValidationError

settings = Settings()
engine = ExtractionEngine(settings)
request_semaphore = asyncio.Semaphore(settings.web_max_concurrent_requests)
static_dir = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.request_temp_root.mkdir(parents=True, exist_ok=True)
    if settings.warm_models_on_startup:
        await anyio.to_thread.run_sync(engine.warm)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Một API duy nhất nhận PDF và trả text/Markdown. "
        "pdf-inspector xử lý trang native; PP-OCRv6 + VietOCR xử lý trang scan."
    ),
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    total = 0
    first_chunk = True
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            if first_chunk:
                first_chunk = False
                if not chunk.startswith(b"%PDF-"):
                    raise HTTPException(
                        status_code=400,
                        detail={"error_code": "invalid_pdf", "message_vi": "Tệp không có magic PDF"},
                    )
            total += len(chunk)
            if total > settings.web_upload_max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={"error_code": "pdf_too_large", "message_vi": "PDF vượt giới hạn upload"},
                )
            target.write(chunk)
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "empty_upload", "message_vi": "Tệp upload rỗng"},
        )
    return total


@app.post(
    "/api/extract",
    response_model=ExtractResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def extract_pdf(file: UploadFile = File(...)) -> ExtractResponse:
    filename = Path(file.filename or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail={"error_code": "invalid_pdf", "message_vi": "Chỉ chấp nhận tệp .pdf"},
        )

    settings.request_temp_root.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        suffix=".pdf",
        prefix="upload-",
        dir=settings.request_temp_root,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        await _save_upload(file, temp_path)
        async with request_semaphore:
            return await anyio.to_thread.run_sync(engine.extract, temp_path, filename)
    except PdfValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": exc.error_code, "message_vi": exc.message_vi},
        ) from exc
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error_code": "model_asset_missing", "message_vi": str(exc)},
        ) from exc
    except Exception as exc:
        engine.logger.exception(
            "Lỗi không mong đợi khi bóc tách PDF: %s",
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail={"error_code": "extraction_failed", "message_vi": str(exc)},
        ) from exc
    finally:
        await file.close()
        temp_path.unlink(missing_ok=True)
