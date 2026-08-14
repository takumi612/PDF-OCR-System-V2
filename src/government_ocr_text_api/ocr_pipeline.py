from __future__ import annotations

import re
import statistics
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .config import Settings
from .line_crops import (
    make_axis_aligned_retry_crop,
    make_horizontal_expanded_line_crop,
    make_line_crop,
)
from .line_detector import PaddleLineDetector
from .logging_utils import measure
from .models import LineCrop, OcrLineResult, PageImage, PageResult, Recognition
from .pdf_session import PdfDocumentSession
from .quality import PageQualityGate, PageQualityResult
from .tesseract_verifier import TesseractPageVerifier, apply_tesseract_verification
from .vietocr_recognizer import VietOcrRecognizer


def _punctuation_signature(value: str) -> tuple[str, ...]:
    return tuple(
        character
        for character in value
        if unicodedata.category(character).startswith("P")
    )


def _structural_heading_labels(settings: Settings) -> tuple[str, ...]:
    labels = tuple(
        label.strip().casefold()
        for label in settings.chapter_heading_retry_labels.split(",")
        if label.strip()
    )
    return labels or ("chương",)


def _is_incomplete_chapter_heading(
    text: str,
    labels: Sequence[str] = ("chương",),
) -> bool:
    cleaned = text.strip().rstrip(".:;-").strip().casefold()
    return cleaned in {label.casefold() for label in labels}



def _nontext_crop_reason(crop: LineCrop, settings: Settings) -> str | None:
    """Return a geometry-only reason when a crop is unsafe to send to seq2seq OCR.

    The filter is deliberately conservative: it only removes nearly blank crops
    or very wide marks whose ink is confined to a thin horizontal band (rules /
    dotted leaders). It never inspects recognized words or document vocabulary.
    """
    if not settings.ocr_nontext_crop_filter_enabled:
        return None
    gray = np.asarray(crop.image.convert("L"), dtype=np.uint8)
    if gray.size == 0 or gray.ndim != 2:
        return "empty_image"
    ink = gray < 210
    ink_ratio = float(ink.mean())
    if ink_ratio <= settings.ocr_nontext_min_ink_ratio:
        return "near_blank"

    height, width = ink.shape
    aspect_ratio = width / max(height, 1)
    # A row counts as active only when it contains a non-trivial amount of ink.
    # This catches padded horizontal rules without relying on the crop's absolute
    # pixel height.
    row_threshold = max(2.0 / max(width, 1), 0.01)
    active_row_ratio = float(np.mean(ink.mean(axis=1) >= row_threshold))
    ink_column_ratio = float(np.mean(np.any(ink, axis=0)))

    binary = ink.astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = [
        stats[index]
        for index in range(1, component_count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= 2
    ]
    component_heights = [int(item[cv2.CC_STAT_HEIGHT]) for item in components]
    component_widths = [int(item[cv2.CC_STAT_WIDTH]) for item in components]
    max_component_height_ratio = max(component_heights, default=0) / max(height, 1)
    max_component_width_ratio = max(component_widths, default=0) / max(width, 1)
    small_component_ratio = (
        float(
            np.mean(
                [
                    component_height / max(height, 1)
                    <= settings.ocr_nontext_rule_max_component_height_ratio
                    for component_height in component_heights
                ]
            )
        )
        if component_heights
        else 0.0
    )

    if aspect_ratio >= settings.ocr_nontext_rule_min_aspect_ratio:
        thin_band = (
            ink_ratio <= settings.ocr_nontext_rule_max_ink_ratio
            and active_row_ratio <= settings.ocr_nontext_rule_max_active_row_ratio
            and ink_column_ratio >= settings.ocr_nontext_rule_min_ink_column_ratio
            and max_component_height_ratio
            <= settings.ocr_nontext_rule_max_component_height_ratio
        )
        solid_rule = (
            max_component_width_ratio >= 0.80
        )
        if thin_band and solid_rule:
            return "horizontal_rule_or_dotted_leader"
    return None


def _same_geometric_line(left: LineCrop, right: LineCrop) -> bool:
    """Only treat equal text as duplicate when detector geometry also overlaps."""
    return left.polygon.bbox.iou(right.polygon.bbox) >= 0.80


def _merge_duplicate_recognitions(
    left: Recognition,
    right: Recognition,
) -> Recognition:
    risk_rank = {"none": 0, "medium": 1, "high": 2}
    higher_risk = (
        right
        if risk_rank[right.semantic_risk] > risk_rank[left.semantic_risk]
        else left
    )
    return Recognition(
        text=left.text,
        confidence=min(left.confidence, right.confidence),
        error_code=left.error_code or right.error_code,
        message_vi=left.message_vi or right.message_vi,
        raw_text=higher_risk.raw_text or left.raw_text or right.raw_text,
        semantic_risk=higher_risk.semantic_risk,
        semantic_reasons=tuple(
            dict.fromkeys((*left.semantic_reasons, *right.semantic_reasons))
        ),
        secondary_confidence=(
            higher_risk.secondary_confidence
            if higher_risk.secondary_confidence is not None
            else left.secondary_confidence or right.secondary_confidence
        ),
        verifier_text=(
            higher_risk.verifier_text or left.verifier_text or right.verifier_text
        ),
        verifier_confidence=(
            higher_risk.verifier_confidence
            if higher_risk.verifier_confidence is not None
            else left.verifier_confidence or right.verifier_confidence
        ),
    )


def _is_complete_chapter_heading(
    text: str,
    labels: Sequence[str] = ("chương",),
) -> bool:
    cleaned = " ".join(text.strip().rstrip(".:;-").split()).casefold()
    label_pattern = "|".join(re.escape(label.casefold()) for label in labels if label)
    if not label_pattern:
        return False
    pattern = re.compile(
        rf"^(?:{label_pattern})\s+(?:[ivxlcdm]{{1,8}}|\d{{1,4}})$",
        re.IGNORECASE,
    )
    return bool(pattern.fullmatch(cleaned))


@dataclass
class PreparedOcrPage:
    page_index: int
    ocr_reason: str | None
    quality: PageQualityResult
    crops: list[LineCrop]
    metrics: dict[str, Any]
    early_result: PageResult | None = None
    page_image: PageImage | None = None


class OcrPagePipeline:
    def __init__(
        self,
        settings: Settings,
        detector: PaddleLineDetector,
        recognizer: VietOcrRecognizer,
        quality_gate: PageQualityGate,
        tesseract_verifier: TesseractPageVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.recognizer = recognizer
        self.quality_gate = quality_gate
        self.tesseract_verifier = tesseract_verifier or TesseractPageVerifier(settings)

    def prepare_page(
        self,
        session: PdfDocumentSession,
        page_index: int,
        ocr_reason: str | None,
    ) -> PreparedOcrPage:
        metrics: dict[str, Any] = {}
        with measure(metrics, "pdf_render_ms"):
            page = session.render_page(page_index)
        with measure(metrics, "quality_gate_ms"):
            quality = self.quality_gate.evaluate(page)
            page = self.quality_gate.apply(page, quality)

        if quality.low_ink:
            reasons = ["empty_ocr_text", "low_ink_page"]
            marker = (
                f"[OCR_SEMANTIC_RISK page={page_index + 1} line=0 "
                f"reasons={','.join(reasons)}]"
            )
            metrics.update(
                {
                    "page_total_ms": round(sum(metrics.values()), 3),
                    "dark_pixel_ratio": quality.dark_pixel_ratio,
                    "estimated_skew": quality.estimated_skew,
                    "quality_action": quality.action,
                }
            )
            result = PageResult(
                page_index=page_index,
                page_number=page_index + 1,
                source="ocr",
                text="",
                markdown="",
                needs_ocr=True,
                ocr_reason=ocr_reason or "low_ink",
                needs_review=True,
                line_count=0,
                metrics=metrics,
                error_codes=["empty_ocr_text"],
                ai_safe_text=marker,
                ai_ready=False,
                semantic_risk_count=1,
                line_results=[
                    OcrLineResult(
                        line_index=0,
                        text="",
                        confidence=0.0,
                        error_code="empty_ocr_text",
                        semantic_risk="high",
                        semantic_reasons=reasons,
                    )
                ],
            )
            return PreparedOcrPage(
                page_index,
                ocr_reason,
                quality,
                [],
                metrics,
                result,
                page,
            )

        with measure(metrics, "line_detection_ms"):
            polygons = self.detector.detect(page)
        metrics["layout_ordering"] = dict(getattr(self.detector, "last_metrics", {}))
        with measure(metrics, "line_crop_ms"):
            raw_crops = [
                make_line_crop(
                    page,
                    polygon,
                    self.settings.line_crop_padding_ratio,
                    f"p{page_index:04d}-l{index:04d}",
                )
                for index, polygon in enumerate(polygons)
            ]
            crops: list[LineCrop] = []
            filter_reasons: dict[str, int] = {}
            for crop in raw_crops:
                reason = _nontext_crop_reason(crop, self.settings)
                if reason is None:
                    crops.append(crop)
                    continue
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            metrics["nontext_crop_filter"] = {
                "enabled": self.settings.ocr_nontext_crop_filter_enabled,
                "input_count": len(raw_crops),
                "kept_count": len(crops),
                "filtered_count": len(raw_crops) - len(crops),
                "reason_counts": filter_reasons,
            }
        return PreparedOcrPage(
            page_index,
            ocr_reason,
            quality,
            crops,
            metrics,
            page_image=page,
        )

    def extract_pages(
        self,
        session: PdfDocumentSession,
        page_specs: Sequence[tuple[int, str | None]],
    ) -> list[PageResult]:
        """Xử lý một cửa sổ trang: detect/crop từng trang, nhận dạng chung một lần."""
        prepared = [
            self.prepare_page(session, page_index, reason)
            for page_index, reason in page_specs
        ]
        active = [page for page in prepared if page.early_result is None]
        flat_crops = [crop for page in active for crop in page.crops]
        offsets: dict[int, tuple[int, int]] = {}
        cursor = 0
        for page in active:
            offsets[page.page_index] = (cursor, cursor + len(page.crops))
            cursor += len(page.crops)

        window_ms = 0.0
        recognitions = []
        verifier_executor: ThreadPoolExecutor | None = None
        verifier_futures: dict[int, Future[Any]] = {}
        supports_overlap = bool(
            hasattr(self.tesseract_verifier, "collect_page")
            and hasattr(self.tesseract_verifier, "verify_collected")
        )
        if flat_crops and supports_overlap:
            verifier_pages = [page for page in active if page.page_image is not None]
            if verifier_pages:
                verifier_executor = ThreadPoolExecutor(
                    max_workers=min(2, len(verifier_pages)),
                    thread_name_prefix="tesseract-page",
                )
                verifier_futures = {
                    page.page_index: verifier_executor.submit(
                        self.tesseract_verifier.collect_page,
                        page.page_image,
                    )
                    for page in verifier_pages
                }
        if flat_crops:
            started = time.perf_counter()
            try:
                recognitions = self.recognizer.recognize(flat_crops)
                window_ms = round((time.perf_counter() - started) * 1000, 3)
            except Exception:
                if verifier_executor is not None:
                    verifier_executor.shutdown(wait=False, cancel_futures=True)
                raise

        recognitions_by_page: dict[int, list[Recognition]] = {}
        metrics_by_page: dict[int, dict[str, Any]] = {}
        for page in active:
            start, end = offsets[page.page_index]
            page_recognitions = list(recognitions[start:end])
            page_metrics = dict(self.recognizer.last_page_metrics.get(page.page_index, {}))
            page_recognitions = self._refine_chapter_headings(
                page,
                page_recognitions,
                page_metrics,
            )
            if page.page_image is not None:
                future = verifier_futures.get(page.page_index)
                if future is not None:
                    lines, collected_metrics = future.result()
                    page_recognitions, verifier_metrics = (
                        self.tesseract_verifier.verify_collected(
                            page.crops,
                            page_recognitions,
                            lines,
                            collected_metrics,
                        )
                    )
                else:
                    page_recognitions, verifier_metrics = self.tesseract_verifier.verify(
                        page.page_image,
                        page.crops,
                        page_recognitions,
                    )
                page.metrics["tesseract_verifier"] = verifier_metrics
                page.metrics["tesseract_verifier_ms"] = float(
                    verifier_metrics.get("elapsed_ms", 0.0)
                )
            recognitions_by_page[page.page_index] = page_recognitions
            metrics_by_page[page.page_index] = page_metrics
        if verifier_executor is not None:
            verifier_executor.shutdown(wait=True)

        recognitions_by_page = self._run_selective_semantic_verification_window(
            active,
            recognitions_by_page,
            metrics_by_page,
        )
        recognitions_by_page = self._run_partial_remediation_window(
            active,
            recognitions_by_page,
            metrics_by_page,
        )

        results: list[PageResult] = []
        for page in prepared:
            if page.early_result is not None:
                results.append(page.early_result)
                continue
            page_recognitions = recognitions_by_page[page.page_index]
            page_metrics = metrics_by_page[page.page_index]
            page.metrics["secondary_semantic_verifier_ms"] = round(
                float(page_metrics.get("semantic_verification_ms", 0.0)),
                3,
            )
            recognition_ms = float(page_metrics.get("recognition_ms_allocated", 0.0))
            chapter_retry_ms = float(page_metrics.get("chapter_heading_retry_ms", 0.0))
            page.metrics["vietocr_ms"] = round(recognition_ms + chapter_retry_ms, 3)
            page.metrics["recognition_window_ms"] = window_ms
            page.metrics["recognition_window_page_count"] = len(active)
            page.metrics["recognition"] = page_metrics
            results.append(self._finalize(page, page_recognitions))
        return results

    def _semantic_candidate_indices(
        self,
        recognitions: Sequence[Recognition],
    ) -> list[int]:
        if not self.settings.tesseract_verifier_enabled:
            return [
                index
                for index, recognition in enumerate(recognitions)
                if recognition.secondary_confidence is None
            ]
        return [
            index
            for index, recognition in enumerate(recognitions)
            if recognition.secondary_confidence is None
            and (
                recognition.error_code is not None
                or recognition.confidence
                < self.settings.semantic_primary_low_confidence
                or recognition.semantic_risk != "none"
                or any(character.isdigit() for character in recognition.text)
                or (
                    recognition.verifier_text is not None
                    and _punctuation_signature(recognition.text)
                    != _punctuation_signature(recognition.verifier_text)
                )
            )
        ]

    @staticmethod
    def _initialize_empty_semantic_metrics(
        metrics: dict[str, Any],
        skipped_count: int,
    ) -> None:
        metrics.setdefault("semantic_candidate_count", 0)
        metrics["semantic_skipped_count"] = (
            metrics.get("semantic_skipped_count", 0) + skipped_count
        )
        metrics.setdefault("semantic_verified_count", 0)
        metrics.setdefault("semantic_secondary_unavailable_count", 0)
        metrics.setdefault("semantic_secondary_error_count", 0)
        metrics.setdefault("semantic_auto_trimmed_count", 0)
        metrics.setdefault("semantic_high_risk_count", 0)
        metrics.setdefault("semantic_verification_ms", 0.0)

    def _run_selective_semantic_verification_window(
        self,
        pages: Sequence[PreparedOcrPage],
        recognitions_by_page: dict[int, list[Recognition]],
        metrics_by_page: dict[int, dict[str, Any]],
    ) -> dict[int, list[Recognition]]:
        if not self.settings.semantic_selective_verification_enabled:
            return recognitions_by_page

        flat_crops: list[LineCrop] = []
        flat_retry_crops: list[LineCrop] = []
        flat_recognitions: list[Recognition] = []
        candidate_indices: list[int] = []
        offsets: dict[int, tuple[int, int]] = {}
        cursor = 0
        for page in pages:
            page_recognitions = recognitions_by_page[page.page_index]
            start = cursor
            flat_crops.extend(page.crops)
            if page.page_image is not None and self.settings.semantic_retry_enabled:
                flat_retry_crops.extend(
                    make_axis_aligned_retry_crop(
                        page.page_image,
                        crop,
                        self.settings.semantic_retry_crop_padding_height_ratio,
                    )
                    for crop in page.crops
                )
            else:
                flat_retry_crops.extend(page.crops)
            flat_recognitions.extend(page_recognitions)
            candidate_indices.extend(
                start + index
                for index in self._semantic_candidate_indices(page_recognitions)
            )
            cursor += len(page_recognitions)
            offsets[page.page_index] = (start, cursor)

        if not candidate_indices:
            for page in pages:
                self._initialize_empty_semantic_metrics(
                    metrics_by_page[page.page_index],
                    len(recognitions_by_page[page.page_index]),
                )
            return recognitions_by_page

        verified = self.recognizer.verify_semantic_candidates(
            flat_crops,
            flat_recognitions,
            candidate_indices,
            metrics_by_page,
            retry_crops=flat_retry_crops,
        )
        for page in pages:
            start, end = offsets[page.page_index]
            recognitions_by_page[page.page_index] = list(verified[start:end])
        return recognitions_by_page

    def _run_selective_semantic_verification(
        self,
        page: PreparedOcrPage,
        recognitions: list[Recognition],
        page_metrics: dict[str, Any],
    ) -> list[Recognition]:
        if not self.settings.semantic_selective_verification_enabled:
            return recognitions
        result = self._run_selective_semantic_verification_window(
            [page],
            {page.page_index: recognitions},
            {page.page_index: page_metrics},
        )
        return result[page.page_index]

    def _run_partial_remediation_window(
        self,
        pages: Sequence[PreparedOcrPage],
        recognitions_by_page: dict[int, list[Recognition]],
        metrics_by_page: dict[int, dict[str, Any]],
    ) -> dict[int, list[Recognition]]:
        """Chạy đúng một lượt cứu hộ có kiểm chứng cho high-risk còn sót."""
        if not self.settings.partial_remediation_enabled:
            return recognitions_by_page
        if not hasattr(self.tesseract_verifier, "recognize_targeted") or not hasattr(
            self.recognizer,
            "remediate_high_risk_candidates",
        ):
            return recognitions_by_page

        selected_crops: list[LineCrop] = []
        selected_recognitions: list[Recognition] = []
        selected_locations: list[tuple[int, int]] = []
        remediation_metrics: dict[int, dict[str, Any]] = {}
        for page in pages:
            page_recognitions = recognitions_by_page[page.page_index]
            candidates = [
                index
                for index, recognition in enumerate(page_recognitions)
                if recognition.semantic_risk == "high" and index < len(page.crops)
            ]
            candidates.sort(
                key=lambda index: (
                    "tesseract_numeric_disagreement"
                    not in page_recognitions[index].semantic_reasons,
                    page_recognitions[index].confidence,
                    index,
                )
            )
            selected = candidates[
                : self.settings.partial_remediation_max_lines_per_page
            ]
            page_metric = {
                "enabled": True,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "attempted_count": 0,
                "applied_count": 0,
                "events": [],
            }
            remediation_metrics[page.page_index] = page_metric
            page.metrics["partial_remediation"] = page_metric
            if not selected or page.page_image is None:
                continue
            retry_crops = [
                make_axis_aligned_retry_crop(
                    page.page_image,
                    page.crops[index],
                    self.settings.semantic_retry_crop_padding_height_ratio,
                )
                for index in selected
            ]
            evidence, verifier_metrics = self.tesseract_verifier.recognize_targeted(
                retry_crops
            )
            page_metric["targeted_tesseract"] = verifier_metrics
            for index, retry_crop, verifier in zip(selected, retry_crops, evidence):
                if verifier is None:
                    continue
                recognition = apply_tesseract_verification(
                    page_recognitions[index],
                    verifier,
                    self.settings,
                )
                selected_crops.append(retry_crop)
                selected_recognitions.append(recognition)
                selected_locations.append((page.page_index, index))

        if selected_crops:
            repaired = self.recognizer.remediate_high_risk_candidates(
                selected_crops,
                selected_recognitions,
                remediation_metrics,
            )
            for (page_index, line_index), recognition in zip(
                selected_locations,
                repaired,
            ):
                recognitions_by_page[page_index][line_index] = recognition

        for page in pages:
            page_index = page.page_index
            page_metric = remediation_metrics[page_index]
            tesseract_ms = float(
                (page_metric.get("targeted_tesseract") or {}).get("elapsed_ms", 0.0)
            )
            recognition_ms = float(page_metric.get("elapsed_ms", 0.0))
            page.metrics["partial_remediation_ms"] = round(
                tesseract_ms + recognition_ms,
                3,
            )
            metrics_by_page[page_index]["semantic_high_risk_count"] = sum(
                recognition.semantic_risk == "high"
                for recognition in recognitions_by_page[page_index]
            )
        return recognitions_by_page

    def _refine_chapter_headings(
        self,
        page: PreparedOcrPage,
        recognitions: list[Recognition],
        page_metrics: dict[str, Any],
    ) -> list[Recognition]:
        """Retry cực hẹp cho dòng chỉ nhận được ``Chương``.

        Detector thực tế có thể bỏ ký tự ``I``/``III`` rất hẹp. Lần retry mở
        crop sang phải nhưng chỉ được phép thay output khi dự đoán mới khớp
        chính xác mẫu tiêu đề chương; mọi kết quả dài hoặc nhiễu đều bị từ chối.
        """
        page_metrics.setdefault("chapter_heading_retry_count", 0)
        page_metrics.setdefault("chapter_heading_retry_accepted_count", 0)
        page_metrics.setdefault("chapter_heading_retry_unresolved_count", 0)
        page_metrics.setdefault("chapter_heading_retry_failed_count", 0)
        page_metrics.setdefault("chapter_heading_retry_ms", 0.0)

        if (
            not self.settings.chapter_heading_retry_enabled
            or self.settings.chapter_heading_max_retries_per_page <= 0
            or page.page_image is None
        ):
            return recognitions

        heading_labels = _structural_heading_labels(self.settings)
        candidates = [
            index
            for index, recognition in enumerate(recognitions)
            if index < len(page.crops)
            and _is_incomplete_chapter_heading(recognition.text, heading_labels)
        ][: self.settings.chapter_heading_max_retries_per_page]
        if not candidates:
            return recognitions

        retry_crops = [
            make_horizontal_expanded_line_crop(
                page.page_image,
                page.crops[index],
                self.settings.chapter_heading_expand_height_ratio,
                f"{page.crops[index].crop_id}-chapter-retry",
            )
            for index in candidates
        ]
        page_metrics["chapter_heading_retry_count"] += len(retry_crops)
        started = time.perf_counter()
        try:
            retries = self.recognizer.recognize_targeted(retry_crops)
        except Exception:
            retries = []
            page_metrics["chapter_heading_retry_failed_count"] += len(retry_crops)
        page_metrics["chapter_heading_retry_ms"] += (
            time.perf_counter() - started
        ) * 1000

        for position, index in enumerate(candidates):
            original = recognitions[index]
            retry = retries[position] if position < len(retries) else None
            confidence_floor = max(
                self.settings.chapter_heading_min_confidence,
                original.confidence - 0.25,
            )
            if (
                retry is not None
                and retry.error_code is None
                and _is_complete_chapter_heading(retry.text, heading_labels)
                and retry.confidence >= confidence_floor
            ):
                recognitions[index] = retry
                page_metrics["chapter_heading_retry_accepted_count"] += 1
                continue

            page_metrics["chapter_heading_retry_unresolved_count"] += 1
            if original.error_code is None:
                recognitions[index] = Recognition(
                    text=original.text,
                    confidence=original.confidence,
                    error_code="chapter_heading_incomplete",
                    message_vi=(
                        "Tiêu đề chương có thể thiếu số La Mã; cần đối chiếu PDF gốc"
                    ),
                    raw_text=original.raw_text,
                    semantic_risk="high",
                    semantic_reasons=tuple(
                        dict.fromkeys(
                            (*original.semantic_reasons, "chapter_heading_incomplete")
                        )
                    ),
                    secondary_confidence=original.secondary_confidence,
                    verifier_text=original.verifier_text,
                    verifier_confidence=original.verifier_confidence,
                )
        page_metrics["chapter_heading_retry_ms"] = round(
            float(page_metrics["chapter_heading_retry_ms"]),
            3,
        )
        return recognitions

    def extract_page(
        self,
        session: PdfDocumentSession,
        page_index: int,
        ocr_reason: str | None,
    ) -> PageResult:
        return self.extract_pages(session, [(page_index, ocr_reason)])[0]

    def _finalize(self, page: PreparedOcrPage, recognitions: Sequence[Any]) -> PageResult:
        merged_recognitions: list[tuple[int, Recognition]] = []
        geometry_duplicate_suppressed_count = 0
        for index, recognition in enumerate(recognitions):
            if (
                recognition.text
                and merged_recognitions
                and merged_recognitions[-1][1].text == recognition.text
                and index < len(page.crops)
                and merged_recognitions[-1][0] < len(page.crops)
                and _same_geometric_line(
                    page.crops[merged_recognitions[-1][0]],
                    page.crops[index],
                )
            ):
                previous_index, previous = merged_recognitions[-1]
                merged_recognitions[-1] = (
                    previous_index,
                    _merge_duplicate_recognitions(previous, recognition),
                )
                geometry_duplicate_suppressed_count += 1
                continue
            merged_recognitions.append((index, recognition))

        errors: list[str] = []
        lines: list[str] = []
        ai_safe_lines: list[str] = []
        line_results: list[OcrLineResult] = []
        confidences: list[float] = []
        review = page.quality.needs_review
        for index, recognition in merged_recognitions:
            if recognition.error_code:
                errors.append(recognition.error_code)
                review = True
            semantic_risk = recognition.semantic_risk
            semantic_reasons = list(recognition.semantic_reasons)
            if not recognition.text and semantic_risk != "high":
                semantic_risk = "high"
                if "empty_primary_text" not in semantic_reasons:
                    semantic_reasons.append("empty_primary_text")
            if semantic_risk != "none":
                review = True
            if semantic_risk == "high":
                reasons = ",".join(semantic_reasons) or "unspecified"
                ai_safe_lines.append(
                    f"[OCR_SEMANTIC_RISK page={page.page_index + 1} "
                    f"line={index + 1} reasons={reasons}]"
                )
            elif recognition.text:
                ai_safe_lines.append(recognition.text)

            crop = page.crops[index] if index < len(page.crops) else None
            bbox = crop.polygon.bbox if crop is not None else None
            line_results.append(
                OcrLineResult(
                    line_index=index,
                    crop_id=crop.crop_id if crop is not None else None,
                    text=recognition.text,
                    raw_text=recognition.raw_text,
                    confidence=max(0.0, min(1.0, recognition.confidence)),
                    error_code=recognition.error_code,
                    semantic_risk=semantic_risk,
                    semantic_reasons=semantic_reasons,
                    secondary_confidence=recognition.secondary_confidence,
                    verifier_text=recognition.verifier_text,
                    verifier_confidence=recognition.verifier_confidence,
                    bbox=(
                        [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                        if bbox is not None
                        else None
                    ),
                )
            )
            if not recognition.text:
                continue
            lines.append(recognition.text)
            confidences.append(recognition.confidence)
            if recognition.confidence < self.settings.review_threshold:
                review = True
        page.metrics["geometry_duplicate_suppressed_count"] = (
            geometry_duplicate_suppressed_count
        )

        if not lines and not line_results:
            reasons = ["empty_ocr_text"]
            ai_safe_lines.append(
                f"[OCR_SEMANTIC_RISK page={page.page_index + 1} line=0 "
                f"reasons={','.join(reasons)}]"
            )
            line_results.append(
                OcrLineResult(
                    line_index=0,
                    text="",
                    confidence=0.0,
                    error_code="empty_ocr_text",
                    semantic_risk="high",
                    semantic_reasons=reasons,
                )
            )

        text = "\n".join(lines).strip()
        ai_safe_text = "\n".join(ai_safe_lines).strip()
        semantic_risk_count = sum(
            line.semantic_risk == "high" for line in line_results
        )
        if semantic_risk_count:
            errors.append("semantic_risk_detected")
        if not text:
            errors.append("empty_ocr_text")
            review = True
        page.metrics["page_total_ms"] = round(
            sum(
                float(value)
                for key, value in page.metrics.items()
                if key.endswith("_ms") and key != "recognition_window_ms"
            ),
            3,
        )
        page.metrics["dark_pixel_ratio"] = page.quality.dark_pixel_ratio
        page.metrics["estimated_skew"] = page.quality.estimated_skew
        page.metrics["quality_action"] = page.quality.action

        return PageResult(
            page_index=page.page_index,
            page_number=page.page_index + 1,
            source="ocr",
            text=text,
            markdown=text,
            needs_ocr=True,
            ocr_reason=page.ocr_reason,
            confidence_mean=round(statistics.fmean(confidences), 4) if confidences else None,
            needs_review=review,
            line_count=len(lines),
            metrics=page.metrics,
            error_codes=sorted(set(errors)),
            ai_safe_text=ai_safe_text,
            ai_ready=semantic_risk_count == 0,
            semantic_risk_count=semantic_risk_count,
            line_results=line_results,
        )
