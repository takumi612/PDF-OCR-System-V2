from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .config import Settings
from .extractor import HybridPdfTextExtractor
from .line_detector import PaddleLineDetector
from .logging_utils import configure_logging, log_event
from .native import PdfInspectorNativeExtractor
from .ocr_pipeline import OcrPagePipeline
from .quality import PageQualityGate
from .vietocr_recognizer import VietOcrRecognizer


def _configure_cpu_runtime(settings: Settings, logger: Any) -> dict[str, Any]:
    """Apply conservative CPU thread tuning before any OCR model is loaded.

    Torch and Paddle use different thread pools. We configure Torch explicitly and
    pass Paddle's thread count to ``TextDetection`` later instead of mutating
    OMP/MKL environment variables globally; this avoids accidental oversubscription.
    """
    metrics: dict[str, Any] = {
        "enabled": bool(settings.cpu_runtime_tuning_enabled),
        "device": settings.device,
        "cpu_count": os.cpu_count(),
        "torch_requested_threads": settings.torch_cpu_threads,
        "torch_requested_interop_threads": settings.torch_interop_threads,
        "paddle_requested_threads": settings.paddle_cpu_threads,
        "paddle_mkldnn_requested": settings.paddle_enable_mkldnn,
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
    }
    if not settings.cpu_runtime_tuning_enabled or settings.paddle_device.startswith("gpu"):
        metrics["applied"] = False
        return metrics

    try:
        import torch

        metrics["torch_threads_before"] = int(torch.get_num_threads())
        metrics["torch_interop_threads_before"] = int(torch.get_num_interop_threads())
        torch.set_num_threads(settings.torch_cpu_threads)
        metrics["torch_threads_after"] = int(torch.get_num_threads())
        try:
            torch.set_num_interop_threads(settings.torch_interop_threads)
            metrics["torch_interop_set_error"] = None
        except RuntimeError as exc:
            # PyTorch permits setting inter-op threads only before parallel work
            # starts. A pre-initialized host process may therefore reject it.
            metrics["torch_interop_set_error"] = type(exc).__name__
        metrics["torch_interop_threads_after"] = int(torch.get_num_interop_threads())
        metrics["applied"] = True
    except Exception as exc:  # pragma: no cover - environment dependent
        metrics["applied"] = False
        metrics["torch_error_type"] = type(exc).__name__
        metrics["torch_error_message"] = str(exc)[:200]

    log_event(
        logger,
        "cpu_runtime_configured",
        "Đã cấu hình runtime CPU cho OCR",
        runtime=metrics,
    )
    return metrics


class ExtractionEngine:
    """Engine persistent: detector và VietOCR được tái sử dụng giữa các request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = configure_logging(settings.log_level, settings.log_format)
        self.runtime_metrics = _configure_cpu_runtime(settings, self.logger)
        self.detector = PaddleLineDetector(settings)
        self.recognizer = VietOcrRecognizer(settings)
        self.quality_gate = PageQualityGate(settings)
        self.native = PdfInspectorNativeExtractor(settings)
        self.ocr_pipeline = OcrPagePipeline(
            settings,
            self.detector,
            self.recognizer,
            self.quality_gate,
        )
        self.extractor = HybridPdfTextExtractor(
            settings,
            self.native,
            self.ocr_pipeline,
            self.logger,
        )
        self._lock = threading.Lock()

    def warm(self) -> None:
        with self._lock:
            log_event(self.logger, "model_warmup_started", "Bắt đầu nạp model OCR")
            self.detector.warm()
            self.recognizer.warm()
            log_event(self.logger, "model_warmup_complete", "Đã nạp model OCR")

    def extract(self, path: Path, filename: str):
        # Paddle/VietOCR không được dùng đồng thời trong cùng process.
        with self._lock:
            return self.extractor.extract(path, filename)
