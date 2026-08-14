from __future__ import annotations

import json
import logging
import os
import time
import warnings
from contextlib import contextmanager
from typing import Any, Iterator


def configure_logging(level: str = "INFO", log_format: str = "compact") -> logging.Logger:
    # These dependency checks are harmless in the packaged CPU runtime. Keep
    # real Paddle/Tesseract/model warnings visible.
    warnings.filterwarnings(
        "ignore",
        message=r"No ccache found\..*",
        category=UserWarning,
        module=r"paddle\.utils\.cpp_extension\.extension_utils",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
        module=r"gdown(?:\..*)?",
    )
    logger = logging.getLogger("government_ocr_text_api")
    logger.setLevel(level.upper())
    setattr(logger, "government_ocr_log_format", log_format)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _seconds(value: Any) -> str:
    try:
        seconds = float(value) / 1000.0
    except (TypeError, ValueError):
        return "?"
    if seconds >= 60.0:
        minutes = int(seconds // 60)
        return f"{minutes}m{seconds - minutes * 60:.0f}s"
    return f"{seconds:.1f}s"


def _short_text(value: Any, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1]}…"


def _retry_location(crop_id: Any) -> str:
    value = str(crop_id or "")
    try:
        page_part, line_part = value.split("-", 1)
        return f"p{int(page_part[1:]) + 1}:l{int(line_part[1:]) + 1}"
    except (TypeError, ValueError, IndexError):
        return value or "?"


def _compact_retry_event(event: dict[str, Any], *, label: str = "Retry") -> str:
    action = "ÁP DỤNG" if event.get("applied") else "GIỮ NGUYÊN"
    selected_width = event.get("selected_width")
    width = selected_width if selected_width is not None else "-"
    confidence = event.get("confidence")
    confidence_part = (
        f" | conf={float(confidence):.3f}" if confidence is not None else ""
    )
    lines = [
        f"[OCR][{label}] {_retry_location(event.get('crop_id'))} | {action} | "
        f"width={width} | {_seconds(event.get('elapsed_ms'))}{confidence_part} | "
        f"{event.get('reason', '?')}",
        f"  trước: \"{_short_text(event.get('before_text'))}\"",
        f"  sau:   \"{_short_text(event.get('after_text'))}\"",
    ]
    return "\n".join(lines)


def _compact_event(payload: dict[str, Any]) -> str:
    event = str(payload.get("event", "event"))
    if event == "document_started":
        return (
            f"[OCR] Bắt đầu | {payload.get('filename', '?')} | "
            f"{payload.get('page_count', '?')} trang"
        )
    if event == "native_routing_complete":
        return (
            "[OCR] Phân tuyến | "
            f"native={payload.get('native_page_count', 0)} | "
            f"ocr={payload.get('ocr_page_count', 0)} | "
            f"{_seconds(payload.get('native_processing_ms'))}"
        )
    if event == "page_complete":
        metrics = payload.get("metrics") or {}
        recognition = metrics.get("recognition") or {}
        tesseract = metrics.get("tesseract_verifier") or {}
        remediation = metrics.get("partial_remediation") or {}
        risk = recognition.get("semantic_high_risk_count", 0)
        retry = recognition.get("semantic_consensus_retry_count", 0)
        retry_attempted = recognition.get("semantic_consensus_retry_attempted_count")
        surface = recognition.get("semantic_surface_consensus_count", 0)
        verifier = recognition.get("semantic_verifier_consensus_count", 0)
        matched = tesseract.get("matched_line_count", 0)
        input_lines = tesseract.get("input_line_count", payload.get("line_count", 0))
        disagreement = tesseract.get("disagreement_line_count", 0)
        review = "CÓ" if payload.get("needs_review") else "KHÔNG"
        page_number = int(payload.get("page_index", 0)) + 1
        retry_summary = (
            f"{retry}/{retry_attempted}"
            if retry_attempted is not None
            else str(retry)
        )
        remediation_summary = ""
        if remediation:
            remediation_summary = (
                f" | cứu hộ={remediation.get('applied_count', 0)}/"
                f"{remediation.get('attempted_count', 0)}"
            )
        lines = [(
            f"[OCR] Trang {page_number}/{payload.get('page_count', '?')} | "
            f"{payload.get('line_count', 0)} dòng | "
            f"{_seconds(metrics.get('page_total_ms'))} | cần duyệt: {review} | "
            f"rủi ro: {risk} | Tesseract: {matched}/{input_lines}, lệch {disagreement} | "
            f"sửa: retry={retry_summary}, dấu={surface}, từ={verifier}"
            f"{remediation_summary}"
        )]
        stage_keys = (
            ("render", "pdf_render_ms"),
            ("detect", "line_detection_ms"),
            ("VietOCR", "vietocr_ms"),
            ("Tesseract", "tesseract_verifier_ms"),
            ("kiểm chứng", "secondary_semantic_verifier_ms"),
            ("cứu hộ", "partial_remediation_ms"),
        )
        if any(metrics.get(key) is not None for _, key in stage_keys):
            stage_text = " | ".join(
                f"{label}={_seconds(metrics.get(key))}" for label, key in stage_keys
            )
            lines.append(f"[OCR][Time] p{page_number} | {stage_text}")
        lines.extend(
            _compact_retry_event(retry_event)
            for retry_event in recognition.get("semantic_consensus_retry_events", ())
        )
        lines.extend(
            _compact_retry_event(retry_event, label="Rescue")
            for retry_event in remediation.get("events", ())
        )
        return "\n".join(lines)
    if event == "document_complete":
        ready = "CÓ" if payload.get("ai_ready") else "KHÔNG"
        return (
            f"[OCR] Hoàn tất | {payload.get('filename', '?')} | "
            f"trạng thái={str(payload.get('status', '?')).upper()} | "
            f"{_seconds(payload.get('processing_time_ms'))} | "
            f"rủi ro={payload.get('semantic_risk_count', 0)} | AI-ready: {ready}"
        )
    return f"[OCR] {payload.get('message_vi', event)}"


def format_event(
    event: str,
    message_vi: str,
    *,
    log_format: str = "compact",
    timestamp: float | None = None,
    pid: int | None = None,
    **fields: Any,
) -> str:
    payload = {
        "event": event,
        "message_vi": message_vi,
        "timestamp": time.time() if timestamp is None else timestamp,
        "pid": os.getpid() if pid is None else pid,
        **fields,
    }
    if log_format == "json":
        return json.dumps(payload, ensure_ascii=False, default=str)
    return _compact_event(payload)


def log_event(logger: logging.Logger, event: str, message_vi: str, **fields: Any) -> None:
    logger.info(
        format_event(
            event,
            message_vi,
            log_format=getattr(logger, "government_ocr_log_format", "compact"),
            **fields,
        )
    )


@contextmanager
def measure(metrics: dict[str, Any], key: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = round((time.perf_counter() - started) * 1000, 3)
