from __future__ import annotations

import contextlib
import difflib
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image, ImageOps

from .config import Settings
from .models import LineCrop, Recognition
from .semantic_guard import evaluate_semantic_line
from .semantic_retry import (
    RetryVariant,
    choose_consensus_retry,
    choose_verifier_consensus,
    normalize_legal_collocations,
    restore_three_engine_separators,
)
from .tesseract_verifier import TesseractLine, apply_tesseract_verification

_PAGE_ID = re.compile(r"^p(?P<page>\d+)-")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_CHAR_RUN = re.compile(r"(?P<char>\w)(?P=char){5,}", re.UNICODE | re.IGNORECASE)
_PUNCT_RUN = re.compile(r"(?:\s*[,.;:_-]\s*){6,}", re.UNICODE)
_LEXICAL_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_REVALIDATABLE_TESSERACT_REASONS = {
    "tesseract_numeric_disagreement",
    "tesseract_diacritic_disagreement",
    "tesseract_material_disagreement",
}
_NON_BLOCKING_CONSENSUS_REASONS = {
    "legal_collocation_normalized",
    "three_engine_separator_consensus_applied",
}


def _semantic_retry_priority(
    recognition: Recognition,
    additional_reasons: Sequence[str] = (),
) -> float:
    reasons = set(recognition.semantic_reasons)
    reasons.update(additional_reasons)
    priority = 0.0
    if "tesseract_numeric_disagreement" in reasons:
        priority += 300.0
    if recognition.error_code == "decoder_loop_trimmed":
        priority += 260.0
    elif recognition.error_code is not None:
        priority += 100.0
    if "secondary_indicates_primary_omission" in reasons:
        priority += 220.0
    if "tesseract_material_disagreement" in reasons:
        priority += 180.0
    if "secondary_material_disagreement" in reasons:
        priority += 160.0
    priority += max(0.0, 1.0 - recognition.confidence) * 10.0
    return priority


def _normalize_recognition_legal_collocations(
    recognition: Recognition,
) -> Recognition:
    normalized = normalize_legal_collocations(recognition.text)
    if normalized == recognition.text:
        return recognition
    reasons = tuple(
        dict.fromkeys((*recognition.semantic_reasons, "legal_collocation_normalized"))
    )
    risk = recognition.semantic_risk
    if risk == "none":
        risk = "medium"
    return Recognition(
        text=normalized,
        confidence=recognition.confidence,
        error_code=recognition.error_code,
        message_vi=recognition.message_vi,
        raw_text=recognition.raw_text or recognition.text,
        semantic_risk=risk,
        semantic_reasons=reasons,
        secondary_confidence=recognition.secondary_confidence,
        verifier_text=recognition.verifier_text,
        verifier_confidence=recognition.verifier_confidence,
    )


def _revalidate_tesseract_after_separator_consensus(
    recognition: Recognition,
    normalized_text: str,
    crop: LineCrop,
    settings: Settings,
) -> Recognition:
    """Drop stale disagreement reasons, then evaluate the repaired text again."""
    had_revalidatable_reason = any(
        reason in _REVALIDATABLE_TESSERACT_REASONS
        for reason in recognition.semantic_reasons
    )
    retained_reasons = tuple(
        reason
        for reason in recognition.semantic_reasons
        if reason not in _REVALIDATABLE_TESSERACT_REASONS
    )
    reasons = tuple(
        dict.fromkeys((*retained_reasons, "three_engine_separator_consensus_applied"))
    )
    blocking_reasons = tuple(
        reason
        for reason in reasons
        if reason not in _NON_BLOCKING_CONSENSUS_REASONS
    )
    base = Recognition(
        text=normalized_text,
        confidence=recognition.confidence,
        error_code=recognition.error_code,
        message_vi=recognition.message_vi,
        raw_text=recognition.raw_text or recognition.text,
        semantic_risk=(
            "high"
            if recognition.semantic_risk == "high" and blocking_reasons
            else "medium"
        ),
        semantic_reasons=reasons,
        secondary_confidence=recognition.secondary_confidence,
        verifier_text=recognition.verifier_text,
        verifier_confidence=recognition.verifier_confidence,
    )
    if (
        not had_revalidatable_reason
        or not recognition.verifier_text
        or recognition.verifier_confidence is None
    ):
        return base
    return apply_tesseract_verification(
        base,
        TesseractLine(
            recognition.verifier_text,
            recognition.verifier_confidence,
            crop.polygon.bbox,
        ),
        settings,
    )


def _prioritize_semantic_retry_indices(
    crops: Sequence[LineCrop],
    recognitions: Sequence[Recognition],
    *,
    candidate_indices: Sequence[int],
    max_per_page: int,
    additional_reasons_by_index: dict[int, Sequence[str]] | None = None,
) -> set[int]:
    if max_per_page <= 0:
        return set()
    by_page: dict[int, list[int]] = {}
    for raw_index in candidate_indices:
        index = int(raw_index)
        if not (0 <= index < min(len(crops), len(recognitions))):
            continue
        page = _page_index(crops[index].crop_id)
        by_page.setdefault(page, []).append(index)
    selected: set[int] = set()
    for indices in by_page.values():
        ranked = sorted(
            indices,
            key=lambda index: (
                _semantic_retry_priority(
                    recognitions[index],
                    (
                        additional_reasons_by_index.get(index, ())
                        if additional_reasons_by_index is not None
                        else ()
                    ),
                ),
                -index,
            ),
            reverse=True,
        )
        selected.update(ranked[:max_per_page])
    return selected


def _profile_add(profile: dict[str, Any] | None, key: str, value: float | int) -> None:
    if profile is None:
        return
    profile[key] = profile.get(key, 0) + value


def _runtime_snapshot() -> dict[str, Any]:
    """Collect low-cost runtime telemetry for performance-regression diagnosis."""
    snapshot: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        snapshot.update(
            {
                "process_rss_mb": round(float(getattr(memory, "rss", 0)) / 1048576.0, 3),
                "process_peak_rss_mb": round(
                    float(getattr(memory, "peak_wset", getattr(memory, "rss", 0))) / 1048576.0,
                    3,
                ),
                "process_num_threads": int(process.num_threads()),
            }
        )
        freq = psutil.cpu_freq()
        if freq is not None:
            snapshot.update(
                {
                    "cpu_freq_current_mhz": round(float(freq.current), 2),
                    "cpu_freq_min_mhz": round(float(freq.min), 2),
                    "cpu_freq_max_mhz": round(float(freq.max), 2),
                }
            )
    except Exception:
        pass
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            snapshot["torch_num_threads"] = int(torch.get_num_threads())
            snapshot["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:
        pass
    return snapshot


def _workload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class _ConsensusDecision:
    text: str
    removed_tokens: tuple[str, ...]
    mode: str
    numeric_only_warning: bool = False


@dataclass(frozen=True)
class _RetryView:
    image: Image.Image
    kind: str
    width_reduction_ratio: float
    leading_blank_ratio: float
    trailing_blank_ratio: float

@dataclass(frozen=True)
class _SplitImage:
    image: Image.Image
    core_left: int
    core_right: int
    source_left: int
    source_right: int
    left_overlap: int
    right_overlap: int
    ink_ratio: float
    ink_columns_ratio: float
    is_tail: bool


@dataclass(frozen=True)
class _Segment:
    original_index: int
    segment_index: int
    page_index: int
    image: Image.Image
    resized_width: int
    was_split: bool
    source_left: int
    source_right: int
    left_overlap: int
    right_overlap: int
    ink_ratio: float
    ink_columns_ratio: float
    is_tail: bool


@dataclass(frozen=True)
class _LoopIssue:
    start_token: int
    ngram_size: int
    repeats: int
    partial_tokens: int = 0

    @property
    def severity(self) -> tuple[int, int, int]:
        return self.repeats, self.partial_tokens, self.ngram_size


def _build_predictor_with_safe_weights(
    Predictor: Any,
    config: dict[str, Any],
    enabled: bool,
) -> Any:
    """Ép VietOCR 0.3.13 nạp state_dict bằng ``weights_only=True``."""
    if not enabled:
        return Predictor(config)

    import torch

    original_load = torch.load

    def safe_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", True)
        return original_load(*args, **kwargs)

    torch.load = safe_load
    try:
        return Predictor(config)
    finally:
        torch.load = original_load


def normalize_vietnamese_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _page_index(crop_id: str) -> int:
    match = _PAGE_ID.match(crop_id)
    return int(match.group("page")) if match else -1


def _predictor_dimensions(predictor: Any, settings: Settings) -> tuple[int, int]:
    config = getattr(predictor, "config", None)
    dataset = config.get("dataset") if isinstance(config, dict) else None
    if not isinstance(dataset, dict):
        return settings.vietocr_image_height, 512
    try:
        return int(dataset["image_height"]), int(dataset["image_max_width"])
    except (KeyError, TypeError, ValueError):
        return settings.vietocr_image_height, 512


def _resized_width(image: Image.Image, image_height: int) -> int:
    return max(10, math.ceil((image_height * image.width / max(image.height, 1)) / 10) * 10)


def _smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _image_ink_stats(image: Image.Image) -> tuple[float, float]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = gray < 210
    if ink.size == 0:
        return 0.0, 0.0
    ink_ratio = float(ink.mean())
    ink_columns_ratio = float(np.any(ink, axis=0).mean()) if ink.ndim == 2 else 0.0
    return ink_ratio, ink_columns_ratio


def _trim_image_to_ink(image: Image.Image, margin: int = 3) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = gray < 225
    if not np.any(mask):
        return image
    _, xs = np.where(mask)
    left = max(0, int(xs.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    if right - left < 4:
        return image
    # Chỉ trim theo X để không làm tăng aspect ratio sau chuẩn hóa height=32.
    return image.crop((left, 0, right, image.height)).convert("RGB")


def _trailing_blank_ratio(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = gray < 225
    if not np.any(mask):
        return 1.0
    xs = np.where(mask)[1]
    last_ink = int(xs.max())
    return max(0.0, (image.width - 1 - last_ink) / max(1, image.width))


def _leading_blank_ratio(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = gray < 225
    if not np.any(mask):
        return 1.0
    xs = np.where(mask)[1]
    first_ink = int(xs.min())
    return max(0.0, first_ink / max(1, image.width))


def _ink_tight_crop(image: Image.Image, margin_ratio: float) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = gray < 225
    if not np.any(mask):
        return image.convert("RGB")
    xs = np.where(mask)[1]
    margin = max(2, int(round(image.height * margin_ratio)))
    left = max(0, int(xs.min()) - margin)
    right = min(image.width, int(xs.max()) + margin + 1)
    if right - left < 4:
        return image.convert("RGB")
    return image.crop((left, 0, right, image.height)).convert("RGB")


def _large_blank_runs(mask: np.ndarray, minimum_width: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
            continue
        if value or start is None:
            continue
        if index - start >= minimum_width:
            runs.append((start, index))
        start = None
    return runs


def _dominant_ink_crop(
    image: Image.Image,
    margin_ratio: float,
    detached_gap_height_ratio: float,
    max_removed_ink_ratio: float,
) -> Image.Image | None:
    """Loại island mực ở biên bị tách khỏi dòng bởi một khoảng trắng rất lớn.

    Không dựa vào từ điển. Chỉ dùng hình học: cluster chính phải giữ phần lớn
    lượng mực, còn island bị bỏ phải nằm ngoài một gap lớn theo chiều cao dòng.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = gray < 220
    if not np.any(ink):
        return None
    column_mass = ink.sum(axis=0).astype(np.float32)
    occupied = column_mass > 0
    first = int(np.flatnonzero(occupied)[0])
    last = int(np.flatnonzero(occupied)[-1])
    if last <= first:
        return None

    minimum_gap = max(6, int(round(image.height * detached_gap_height_ratio)))
    gaps = _large_blank_runs(~occupied[first : last + 1], minimum_gap)
    if not gaps:
        return None
    absolute_gaps = [(first + left, first + right) for left, right in gaps]
    boundaries = [first] + [value for gap in absolute_gaps for value in gap] + [last + 1]
    clusters: list[tuple[int, int, float]] = []
    cursor = first
    for gap_left, gap_right in absolute_gaps:
        if gap_left > cursor:
            clusters.append((cursor, gap_left, float(column_mass[cursor:gap_left].sum())))
        cursor = gap_right
    if cursor < last + 1:
        clusters.append((cursor, last + 1, float(column_mass[cursor:last + 1].sum())))
    clusters = [item for item in clusters if item[2] > 0]
    if len(clusters) < 2:
        return None

    main_left, main_right, main_mass = max(
        clusters, key=lambda item: (item[2], item[1] - item[0])
    )
    total_mass = float(column_mass.sum())
    removed_ratio = max(0.0, 1.0 - main_mass / max(total_mass, 1.0))
    main_width_ratio = (main_right - main_left) / max(1, last + 1 - first)
    if removed_ratio > max_removed_ink_ratio or main_width_ratio < 0.45:
        return None

    margin = max(2, int(round(image.height * margin_ratio)))
    left = max(0, main_left - margin)
    right = min(image.width, main_right + margin)
    if right - left >= image.width * 0.96:
        return None
    return image.crop((left, 0, right, image.height)).convert("RGB")


def _build_retry_view(image: Image.Image, settings: Settings) -> _RetryView | None:
    dominant = _dominant_ink_crop(
        image,
        settings.hallucination_guard_margin_height_ratio,
        settings.hallucination_guard_detached_gap_height_ratio,
        settings.hallucination_guard_max_removed_ink_ratio,
    )
    tight = _ink_tight_crop(
        image, settings.hallucination_guard_margin_height_ratio
    )
    candidates: list[tuple[str, Image.Image]] = []
    if dominant is not None:
        candidates.append(("dominant_ink", dominant))
    if tight.width < image.width:
        candidates.append(("tight_ink", tight))
    if not candidates:
        return None
    kind, retry = min(candidates, key=lambda item: item[1].width)
    reduction = max(0.0, 1.0 - retry.width / max(1, image.width))
    if reduction < settings.hallucination_guard_min_crop_change_ratio:
        return None
    return _RetryView(
        image=retry,
        kind=kind,
        width_reduction_ratio=reduction,
        leading_blank_ratio=_leading_blank_ratio(image),
        trailing_blank_ratio=_trailing_blank_ratio(image),
    )


def _lexical_tokens(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in _LEXICAL_TOKEN.finditer(text)]


def _consensus_delete_only(
    original: str,
    retry: str,
    *,
    original_confidence: float,
    retry_confidence: float,
    leading_blank_ratio: float,
    trailing_blank_ratio: float,
    width_reduction_ratio: float,
    settings: Settings,
) -> _ConsensusDecision | None:
    """Chỉ xóa token mà pass ảnh thứ hai không xác nhận.

    Hàm không thay thế hay thêm token. Vì vậy nó không thể tự viết lại câu theo
    ngôn ngữ; mọi thay đổi phải là một subsequence có anchor hình học hai phía.
    """
    original = normalize_vietnamese_text(original)
    retry = normalize_vietnamese_text(retry)
    if not original or not retry or original == retry:
        return None
    original_tokens = _lexical_tokens(original)
    retry_tokens = _lexical_tokens(retry)
    if not original_tokens or not retry_tokens:
        return None
    a = [token[0].casefold() for token in original_tokens]
    b = [token[0].casefold() for token in retry_tokens]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    removed_indices: list[int] = []
    equal_blocks: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            equal_blocks.append((i1, i2))
            continue
        if tag != "delete":
            return None
        removed_indices.extend(range(i1, i2))
    if not removed_indices:
        return None

    removed_fraction = len(removed_indices) / max(1, len(original_tokens))
    if removed_fraction > settings.hallucination_guard_max_removed_token_ratio:
        return None
    if retry_confidence < original_confidence - settings.hallucination_guard_confidence_tolerance:
        return None

    first_removed = min(removed_indices)
    last_removed = max(removed_indices)
    has_left_anchor = any(
        end <= first_removed and end - start >= settings.hallucination_guard_min_anchor_tokens
        for start, end in equal_blocks
    )
    has_right_anchor = any(
        start > last_removed and end - start >= settings.hallucination_guard_min_anchor_tokens
        for start, end in equal_blocks
    )
    if first_removed == 0:
        mode = "prefix"
        if not has_right_anchor or leading_blank_ratio < settings.hallucination_guard_edge_blank_ratio:
            return None
    elif last_removed == len(original_tokens) - 1:
        mode = "suffix"
        if not has_left_anchor or trailing_blank_ratio < settings.hallucination_guard_edge_blank_ratio:
            return None
    else:
        mode = "midline"
        if not settings.hallucination_guard_midline_enabled:
            return None
        if not (has_left_anchor and has_right_anchor):
            return None
        if retry_confidence + settings.hallucination_guard_midline_confidence_bonus < original_confidence:
            return None

    if settings.hallucination_guard_suffix_only and mode != "suffix":
        return None

    removed_tokens = tuple(original_tokens[index][0] for index in removed_indices)
    has_numeric = any(any(char.isdigit() for char in token) for token in removed_tokens)
    if has_numeric and (
        width_reduction_ratio
        < settings.hallucination_guard_numeric_min_crop_change_ratio
        or retry_confidence < original_confidence
    ):
        return None
    if has_numeric:
        return _ConsensusDecision(
            original,
            removed_tokens,
            mode,
            numeric_only_warning=True,
        )

    removed_set = set(removed_indices)
    pieces: list[str] = []
    cursor = 0
    for index, (_, start, end) in enumerate(original_tokens):
        if index not in removed_set:
            pieces.append(original[cursor:end])
        else:
            pieces.append(original[cursor:start])
        cursor = end
    pieces.append(original[cursor:])
    validated = normalize_vietnamese_text("".join(pieces))
    validated = re.sub(r"\s+([,.;:!?])", r"\1", validated)
    validated = re.sub(r"([([{])\s+", r"\1", validated)
    validated = re.sub(r"\s+([)\]}])", r"\1", validated)
    validated = validated.strip(" ,;:-")
    if not validated or validated == original:
        return None
    return _ConsensusDecision(validated, removed_tokens, mode)


def _hallucination_risk(
    segment: _Segment,
    text: str,
    probability: float,
    retry_view: _RetryView | None,
    settings: Settings,
) -> float:
    if not text or retry_view is None:
        return 0.0
    score = 0.0
    if segment.was_split:
        score += 0.25
    if segment.is_tail or segment.segment_index == 0:
        score += 0.08
    score += min(0.25, retry_view.width_reduction_ratio * 0.8)
    if retry_view.kind == "dominant_ink":
        score += 0.20
    if probability < settings.hallucination_guard_min_confidence:
        score += min(0.25, settings.hallucination_guard_min_confidence - probability)
    if segment.ink_ratio < settings.hallucination_guard_low_ink_ratio:
        score += 0.12
    edge_blank = max(retry_view.leading_blank_ratio, retry_view.trailing_blank_ratio)
    if edge_blank >= settings.hallucination_guard_edge_blank_ratio:
        score += min(0.22, edge_blank * 0.5)
    visible_columns = max(1.0, segment.ink_columns_ratio * segment.image.width)
    chars_per_column = len(re.sub(r"\s+", "", text)) / visible_columns
    if chars_per_column > settings.hallucination_guard_chars_per_ink_column:
        score += 0.18
    return min(1.0, score)


@dataclass(frozen=True)
class _DecoderEvidenceTrace:
    raw_text: str
    token_probabilities: tuple[float, ...]
    attention_centers: tuple[float, ...]
    attention_spreads: tuple[float, ...]
    ink_support: tuple[float, ...]
    last_ink_position: float
    novel_ink_support: tuple[float, ...] = ()
    visual_coverage_gain: tuple[float, ...] = ()
    reused_attention_ratio: tuple[float, ...] = ()


@dataclass(frozen=True)
class _DecoderEvidenceTraceOutcome:
    trace: _DecoderEvidenceTrace | None
    error_type: str | None = None
    error_message: str | None = None
    disabled_reason: str | None = None
    fatal: bool = False


class _DecoderContractError(RuntimeError):
    """Raised when the loaded VietOCR decoder does not match seq2seq contract."""


@dataclass(frozen=True)
class _SecondaryRecognition:
    text: str
    confidence: float


def _secondary_prefers_deletion(
    raw_text: str,
    proposed_text: str,
    secondary_text: str,
    margin: float,
) -> tuple[str, float, float]:
    """Compare an independent visual recognizer without letting it rewrite text."""
    raw = normalize_vietnamese_text(raw_text).casefold()
    proposed = normalize_vietnamese_text(proposed_text).casefold()
    secondary = normalize_vietnamese_text(secondary_text).casefold()
    if not raw or not proposed or not secondary:
        return "unavailable", 0.0, 0.0
    raw_ratio = difflib.SequenceMatcher(a=raw, b=secondary, autojunk=False).ratio()
    proposed_ratio = difflib.SequenceMatcher(
        a=proposed, b=secondary, autojunk=False
    ).ratio()
    if proposed_ratio >= raw_ratio + margin:
        return "primary_extra", proposed_ratio, raw_ratio
    if raw_ratio >= proposed_ratio + margin:
        return "conflict", proposed_ratio, raw_ratio
    return "ambiguous", proposed_ratio, raw_ratio


@dataclass(frozen=True)
class _DecoderEvidenceDecision:
    text: str
    removed_text: str
    span_start: int
    span_end: int
    reason: str
    span_kind: str = "suffix"
    numeric_only_warning: bool = False
    expanded_word_count: int = 0
    probability_threshold: float = 0.0
    ink_support_threshold: float = 0.0
    span_probability_mean: float = 0.0
    span_ink_support_mean: float = 0.0
    span_attention_range: float = 0.0
    span_attention_progress: float = 0.0
    left_anchor_probability_mean: float = 0.0
    left_anchor_ink_support_mean: float = 0.0
    left_anchor_attention_mean: float = 0.0
    right_anchor_probability_mean: float = 0.0
    right_anchor_ink_support_mean: float = 0.0
    right_anchor_attention_mean: float = 0.0
    span_visual_coverage_gain_mean: float = 0.0
    span_reused_attention_ratio_mean: float = 0.0
    left_anchor_visual_coverage_gain_mean: float = 0.0
    right_anchor_visual_coverage_gain_mean: float = 0.0
    global_span_start: float = 0.0
    global_span_end: float = 0.0
    right_anchor_global_attention_mean: float = 0.0
    failed_conditions: tuple[str, ...] = ()

    @property
    def trim_start(self) -> int:
        """Compatibility alias retained for callers from the suffix-only guard."""
        return self.span_start


def _normalized_ink_profile(image: Image.Image, bins: int) -> tuple[np.ndarray, float]:
    """Chiếu mật độ mực của ảnh về đúng chiều dài encoder attention."""
    if bins <= 0:
        return np.zeros(0, dtype=np.float32), 1.0
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = (gray < 210).mean(axis=0).astype(np.float32)
    if ink.size == 0 or float(ink.max(initial=0.0)) <= 0.0:
        return np.zeros(bins, dtype=np.float32), 1.0
    edges = np.linspace(0, ink.size, bins + 1, dtype=np.int32)
    profile = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        left, right = int(edges[index]), int(edges[index + 1])
        if right <= left:
            right = min(ink.size, left + 1)
        profile[index] = float(ink[left:right].mean()) if right > left else 0.0
    scale = max(float(np.percentile(profile, 90)), float(profile.max()), 1e-6)
    profile = np.clip(profile / scale, 0.0, 1.0)
    occupied = np.flatnonzero(profile >= 0.05)
    last_ink = float(occupied[-1] / max(1, bins - 1)) if occupied.size else 1.0
    return profile, last_ink


def _visual_grounding_step(
    attention_row: np.ndarray,
    ink_profile: np.ndarray,
    cumulative_attention: np.ndarray,
) -> tuple[float, float, float]:
    """Scale-invariant visual novelty/reuse from continuous attention overlap.

    The previous implementation thresholded individual attention bins at 0.08.
    Real seq2seq attention is often diffuse, so no bin crossed that threshold and
    reuse collapsed to zero. This version never thresholds a bin: it measures
    overlap directly between the current normalized distribution and historical
    coverage, making the metric stable across different encoder lengths.
    """
    if attention_row.size == 0 or ink_profile.size != attention_row.size:
        return 0.0, 0.0, 0.0
    current = np.maximum(np.asarray(attention_row, dtype=np.float32), 0.0)
    mass = float(current.sum())
    if mass <= 0.0:
        return 0.0, 0.0, 0.0
    current = current / mass
    previous = np.maximum(np.asarray(cumulative_attention, dtype=np.float32), 0.0)
    if previous.shape != current.shape:
        return 0.0, 0.0, 0.0

    # Continuous overlap is bounded by current.sum()==1 even though historical
    # coverage is a per-position maximum and does not itself sum to one.
    reused_ratio = float(np.minimum(current, previous).sum())
    novel_attention = np.maximum(current - previous, 0.0)
    visual_mass = current * ink_profile
    total_visual = float(visual_mass.sum())
    incremental_visual = novel_attention * ink_profile
    coverage_gain = float(incremental_visual.sum())
    novel_ratio = coverage_gain / max(total_visual, 1e-8) if total_visual > 0.0 else float(novel_attention.sum())
    return (
        float(np.clip(novel_ratio, 0.0, 1.0)),
        max(0.0, coverage_gain),
        float(np.clip(reused_ratio, 0.0, 1.0)),
    )


def _trace_metric_array(
    trace: _DecoderEvidenceTrace,
    name: str,
    count: int,
    fallback: float,
) -> np.ndarray:
    values = tuple(getattr(trace, name, ()) or ())
    if len(values) >= count:
        return np.asarray(values[:count], dtype=np.float32)
    return np.full(count, fallback, dtype=np.float32)


def _near_repetitive_suffix_start(
    text: str,
    similarity_threshold: float,
    minimum_tokens: int,
) -> int | None:
    """Tìm suffix lặp gần đúng mà không phụ thuộc vào một từ/cụm cụ thể."""
    matches = list(_TOKEN.finditer(text))
    if len(matches) < minimum_tokens:
        return None
    tokens = [match.group(0).casefold() for match in matches]
    earliest: int | None = None
    max_window = min(6, len(tokens) // 2)
    for window in range(2, max_window + 1):
        left = tokens[-2 * window : -window]
        right = tokens[-window:]
        ratio = difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
        if ratio >= similarity_threshold:
            earliest = len(tokens) - 2 * window
            break

    # Bắt vòng lặp biến thể: cùng bigram xuất hiện >=3 lần trong suffix ngắn,
    # dù các từ chen giữa thay đổi nhẹ nên exact n-gram detector bỏ sót.
    search_start = max(2, len(tokens) - 28)
    tail = tokens[search_start:]
    bigram_counts: dict[tuple[str, str], int] = {}
    for first, second in zip(tail, tail[1:]):
        key = (first, second)
        bigram_counts[key] = bigram_counts.get(key, 0) + 1
    if bigram_counts and max(bigram_counts.values()) >= 3:
        unique_ratio = len(set(tail)) / max(1, len(tail))
        if unique_ratio <= 0.72:
            repeated_bigram = max(bigram_counts, key=bigram_counts.get)
            first_local = next(
                index
                for index, pair in enumerate(zip(tail, tail[1:]))
                if pair == repeated_bigram
            )
            repeated_start = max(2, search_start + first_local - 1)
            earliest = repeated_start if earliest is None else min(earliest, repeated_start)

    if earliest is None or len(tokens) - earliest < minimum_tokens:
        return None
    return matches[earliest].start()


def _decoder_evidence_candidate_score(
    segment: _Segment,
    text: str,
    probability: float,
    settings: Settings,
) -> float:
    if not text:
        return 0.0
    score = 0.0
    if segment.was_split:
        score += 0.22
    if segment.is_tail:
        score += 0.12
    if probability < settings.decoder_evidence_candidate_confidence:
        score += min(0.28, settings.decoder_evidence_candidate_confidence - probability + 0.08)
    if _find_decoder_loop(text) is not None or _find_character_loop(text) is not None:
        score += 0.42
    if _near_repetitive_suffix_start(
        text,
        settings.decoder_evidence_near_loop_similarity,
        settings.decoder_evidence_near_loop_min_tokens,
    ) is not None:
        score += 0.40
    if _trailing_blank_ratio(segment.image) >= 0.12:
        score += 0.10
    visible_columns = max(1.0, segment.ink_columns_ratio * segment.image.width)
    if len(re.sub(r"\s+", "", text)) / visible_columns > 0.20:
        score += 0.18
    return min(1.0, score)


def _vocab_token_text(vocab: Any, token_id: int) -> str:
    mapping = getattr(vocab, "i2c", None)
    if isinstance(mapping, dict):
        return str(mapping.get(token_id, ""))
    if isinstance(mapping, (list, tuple)) and 0 <= token_id < len(mapping):
        return str(mapping[token_id])
    return ""


def _attention_matrix(attention: Any, batch_size: int, source_length: int) -> np.ndarray:
    """Normalize decoder attention to ``[batch, source_length]``.

    VietOCR 0.3.13 normally returns ``[B, S]``. Some wrappers preserve a
    singleton target dimension and return ``[B, 1, S]``. Both contracts are
    accepted; any other shape is rejected instead of being guessed.
    """
    array = attention.detach().float().cpu().numpy()
    if array.ndim == 2 and array.shape[0] == batch_size:
        rows = array
    elif array.ndim == 3 and array.shape[0] == batch_size and array.shape[1] == 1:
        rows = array[:, 0, :]
    else:
        raise _DecoderContractError(
            f"unsupported attention shape {tuple(array.shape)}; "
            f"expected ({batch_size}, S) or ({batch_size}, 1, S)"
        )
    if rows.ndim != 2 or rows.shape[1] != source_length:
        raise _DecoderContractError(
            "attention source length "
            f"{rows.shape[1] if rows.ndim == 2 else 'unknown'} does not match "
            f"encoder source length {source_length}"
        )
    rows = np.maximum(rows.astype(np.float32, copy=False), 0.0)
    masses = rows.sum(axis=1, keepdims=True)
    empty = masses[:, 0] <= 0.0
    if np.any(empty):
        rows[empty] = 1.0 / max(1, source_length)
        masses = rows.sum(axis=1, keepdims=True)
    return rows / np.maximum(masses, 1e-12)


def _attention_vector(attention: Any, source_length: int) -> np.ndarray:
    """Backward-compatible single-image attention normalizer."""
    return _attention_matrix(attention, 1, source_length)[0]


def _vocab_special_token_id(
    vocab: Any,
    attribute_names: Sequence[str],
    symbol_names: Sequence[str],
    default: int,
) -> int:
    """Resolve special-token IDs from the loaded vocabulary when available."""
    for name in attribute_names:
        value = getattr(vocab, name, None)
        if isinstance(value, int):
            return value
    mapping = getattr(vocab, "c2i", None)
    if isinstance(mapping, dict):
        for symbol in symbol_names:
            value = mapping.get(symbol)
            if isinstance(value, int):
                return value
    return default


def _trace_seq2seq_attention_batch_detailed(
    predictor: Any,
    images: Sequence[Image.Image],
    settings: Settings,
    profile: dict[str, Any] | None = None,
) -> list[_DecoderEvidenceTraceOutcome]:
    """Trace a same-size image batch with the real VietOCR seq2seq contract.

    This is still an evidence pass, but it avoids one encoder/decoder invocation
    per candidate. Images are grouped by exact inference size by the caller so
    preprocessing remains identical to the primary greedy batch.
    """
    if not images:
        return []
    model = getattr(predictor, "model", None)
    vocab = getattr(predictor, "vocab", None)
    config = getattr(predictor, "config", None)
    transformer = getattr(model, "transformer", None)
    decoder = getattr(transformer, "decoder", None)

    def unavailable(reason: str) -> list[_DecoderEvidenceTraceOutcome]:
        outcome = _DecoderEvidenceTraceOutcome(
            trace=None, disabled_reason=reason, fatal=True
        )
        return [outcome for _ in images]

    if model is None or vocab is None or not isinstance(config, dict):
        return unavailable("predictor_missing_model_vocab_or_config")
    if transformer is None or decoder is None or not callable(decoder):
        return unavailable("unsupported_transformer_decoder")
    if not hasattr(transformer, "forward_encoder") or not hasattr(model, "cnn"):
        return unavailable("unsupported_seq2seq_encoder_contract")
    if not hasattr(vocab, "i2c"):
        return unavailable("unsupported_vocab_contract")

    try:
        import torch
        from torch.nn.functional import softmax
        from vietocr.tool.translate import process_input

        dataset = config.get("dataset", {})
        required_dataset_keys = ("image_height", "image_min_width", "image_max_width")
        missing = [key for key in required_dataset_keys if key not in dataset]
        if missing:
            raise _DecoderContractError(
                "missing dataset config keys: " + ", ".join(missing)
            )
        device = config.get("device", "cpu")
        preprocess_wall = time.perf_counter()
        preprocess_cpu = time.process_time()
        tensors = [
            process_input(
                image,
                int(dataset["image_height"]),
                int(dataset["image_min_width"]),
                int(dataset["image_max_width"]),
            )
            for image in images
        ]
        _profile_add(profile, "decoder_preprocess_wall_ms", (time.perf_counter() - preprocess_wall) * 1000.0)
        _profile_add(profile, "decoder_preprocess_cpu_ms", (time.process_time() - preprocess_cpu) * 1000.0)
        _profile_add(profile, "decoder_trace_input_pixel_count", sum(image.width * image.height for image in images))
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise _DecoderContractError(
                "batch trace requires identical preprocessed tensor shapes; "
                f"got {sorted(shapes)}"
            )
        tensor = torch.cat(tensors, dim=0).to(device)
        batch_size = len(images)
        model.eval()
        with torch.inference_mode():
            encoder_wall = time.perf_counter()
            encoder_cpu = time.process_time()
            src = model.cnn(tensor)
            memory = transformer.forward_encoder(src)
            _profile_add(profile, "decoder_encoder_wall_ms", (time.perf_counter() - encoder_wall) * 1000.0)
            _profile_add(profile, "decoder_encoder_cpu_ms", (time.process_time() - encoder_cpu) * 1000.0)
            if not isinstance(memory, (tuple, list)) or len(memory) != 2:
                raise _DecoderContractError(
                    "forward_encoder must return (hidden, encoder_outputs)"
                )
            hidden, encoder_outputs = memory
            if getattr(hidden, "ndim", None) != 2:
                raise _DecoderContractError(
                    f"hidden shape must be [batch, hidden], got {getattr(hidden, 'shape', None)}"
                )
            if getattr(encoder_outputs, "ndim", None) != 3:
                raise _DecoderContractError(
                    "encoder_outputs shape must be [source, batch, hidden]"
                )
            if int(hidden.shape[0]) != batch_size or int(encoder_outputs.shape[1]) != batch_size:
                raise _DecoderContractError(
                    "encoder batch size does not match trace image count"
                )

            sos_id = _vocab_special_token_id(
                vocab, ("sos", "sos_id"), ("<s>", "<sos>"), 1
            )
            eos_id = _vocab_special_token_id(
                vocab, ("eos", "eos_id"), ("</s>", "<eos>"), 2
            )
            pad_id = _vocab_special_token_id(
                vocab, ("pad", "pad_id"), ("<pad>",), 0
            )
            unk_id = _vocab_special_token_id(
                vocab, ("unk", "unk_id"), ("<unk>",), 3
            )
            special_ids = {pad_id, sos_id, eos_id, unk_id}
            input_token = torch.full(
                (batch_size,), sos_id, dtype=torch.long, device=device
            )
            finished = np.zeros(batch_size, dtype=bool)
            chars: list[list[str]] = [[] for _ in images]
            probabilities: list[list[float]] = [[] for _ in images]
            centers: list[list[float]] = [[] for _ in images]
            spreads: list[list[float]] = [[] for _ in images]
            supports: list[list[float]] = [[] for _ in images]
            novel_supports: list[list[float]] = [[] for _ in images]
            coverage_gains: list[list[float]] = [[] for _ in images]
            reused_ratios: list[list[float]] = [[] for _ in images]
            source_length = int(encoder_outputs.shape[0])
            profiles = [
                _normalized_ink_profile(image, source_length) for image in images
            ]
            positions = np.linspace(
                0.0, 1.0, source_length, dtype=np.float32
            )
            cumulative_attention = [
                np.zeros(source_length, dtype=np.float32) for _ in images
            ]

            for _ in range(settings.decoder_evidence_max_decode_steps):
                decoder_wall = time.perf_counter()
                decoder_cpu = time.process_time()
                result = decoder(input_token, hidden, encoder_outputs)
                _profile_add(profile, "decoder_model_wall_ms", (time.perf_counter() - decoder_wall) * 1000.0)
                _profile_add(profile, "decoder_model_cpu_ms", (time.process_time() - decoder_cpu) * 1000.0)
                _profile_add(profile, "decoder_forward_call_count", 1)
                _profile_add(profile, "decoder_sample_step_count", batch_size)
                _profile_add(profile, "decoder_attention_element_count", batch_size * source_length)
                if not isinstance(result, (tuple, list)) or len(result) != 3:
                    raise _DecoderContractError(
                        "seq2seq decoder must return (prediction, hidden, attention)"
                    )
                prediction, next_hidden, attention = result
                if (
                    getattr(prediction, "ndim", None) != 2
                    or int(prediction.shape[0]) != batch_size
                ):
                    raise _DecoderContractError(
                        "prediction shape must be [batch, vocab], got "
                        f"{getattr(prediction, 'shape', None)}"
                    )
                if getattr(next_hidden, "shape", None) != getattr(hidden, "shape", None):
                    raise _DecoderContractError(
                        "decoder hidden shape changed unexpectedly"
                    )
                attention_extract_wall = time.perf_counter()
                attention_extract_cpu = time.process_time()
                attention_rows = _attention_matrix(
                    attention, batch_size, source_length
                )
                _profile_add(profile, "decoder_attention_extract_wall_ms", (time.perf_counter() - attention_extract_wall) * 1000.0)
                _profile_add(profile, "decoder_attention_extract_cpu_ms", (time.process_time() - attention_extract_cpu) * 1000.0)
                torch_post_wall = time.perf_counter()
                torch_post_cpu = time.process_time()
                distribution = softmax(prediction, dim=-1)
                values, indices = torch.max(distribution, dim=-1)
                _profile_add(profile, "decoder_torch_postprocess_wall_ms", (time.perf_counter() - torch_post_wall) * 1000.0)
                _profile_add(profile, "decoder_torch_postprocess_cpu_ms", (time.process_time() - torch_post_cpu) * 1000.0)
                hidden = next_hidden
                input_token = indices.to(device=device, dtype=torch.long)

                token_ids = indices.detach().cpu().tolist()
                token_probs = values.detach().cpu().tolist()
                for sample_index, (token_id, token_probability) in enumerate(
                    zip(token_ids, token_probs, strict=True)
                ):
                    if finished[sample_index]:
                        continue
                    if int(token_id) == eos_id:
                        finished[sample_index] = True
                        continue
                    if int(token_id) in special_ids:
                        continue
                    attention_row = attention_rows[sample_index]
                    ink_profile, _ = profiles[sample_index]
                    center = float(np.sum(attention_row * positions))
                    spread = float(
                        np.sqrt(
                            np.sum(attention_row * (positions - center) ** 2)
                        )
                    )
                    support = float(np.sum(attention_row * ink_profile))
                    if settings.decoder_evidence_visual_grounding_enabled:
                        grounding_wall = time.perf_counter()
                        grounding_cpu = time.process_time()
                        novel_support, coverage_gain, reused_ratio = _visual_grounding_step(
                            attention_row, ink_profile, cumulative_attention[sample_index]
                        )
                        cumulative_attention[sample_index] = np.maximum(
                            cumulative_attention[sample_index], attention_row
                        )
                        _profile_add(profile, "decoder_visual_grounding_wall_ms", (time.perf_counter() - grounding_wall) * 1000.0)
                        _profile_add(profile, "decoder_visual_grounding_cpu_ms", (time.process_time() - grounding_cpu) * 1000.0)
                    else:
                        novel_support, coverage_gain, reused_ratio = 0.0, 0.0, 0.0
                    token_text = _vocab_token_text(vocab, int(token_id))
                    if not token_text:
                        raise _DecoderContractError(
                            f"vocab has no text for token id {token_id}"
                        )
                    chars[sample_index].append(token_text)
                    char_count = len(token_text)
                    probabilities[sample_index].extend(
                        [float(token_probability)] * char_count
                    )
                    centers[sample_index].extend([center] * char_count)
                    spreads[sample_index].extend([spread] * char_count)
                    supports[sample_index].extend([support] * char_count)
                    # novel_ink_support keeps the absolute incremental ink mass;
                    # visual_coverage_gain is the normalized fraction of newly
                    # consumed visual evidence, which is comparable across crops.
                    novel_supports[sample_index].extend([coverage_gain] * char_count)
                    coverage_gains[sample_index].extend([novel_support] * char_count)
                    reused_ratios[sample_index].extend([reused_ratio] * char_count)
                if bool(np.all(finished)):
                    break

        trace_build_wall = time.perf_counter()
        trace_build_cpu = time.process_time()
        outcomes: list[_DecoderEvidenceTraceOutcome] = []
        for sample_index in range(batch_size):
            _, last_ink = profiles[sample_index]
            outcomes.append(
                _DecoderEvidenceTraceOutcome(
                    trace=_DecoderEvidenceTrace(
                        raw_text="".join(chars[sample_index]),
                        token_probabilities=tuple(probabilities[sample_index]),
                        attention_centers=tuple(centers[sample_index]),
                        attention_spreads=tuple(spreads[sample_index]),
                        ink_support=tuple(supports[sample_index]),
                        last_ink_position=last_ink,
                        novel_ink_support=tuple(novel_supports[sample_index]),
                        visual_coverage_gain=tuple(coverage_gains[sample_index]),
                        reused_attention_ratio=tuple(reused_ratios[sample_index]),
                    )
                )
            )
        _profile_add(profile, "decoder_trace_build_wall_ms", (time.perf_counter() - trace_build_wall) * 1000.0)
        _profile_add(profile, "decoder_trace_build_cpu_ms", (time.process_time() - trace_build_cpu) * 1000.0)
        _profile_add(profile, "decoder_trace_character_count", sum(len(outcome.trace.raw_text) for outcome in outcomes if outcome.trace is not None))
        return outcomes
    except Exception as exc:
        outcome = _DecoderEvidenceTraceOutcome(
            trace=None,
            error_type=type(exc).__name__,
            error_message=str(exc)[:300],
            fatal=True,
        )
        return [outcome for _ in images]

def _trace_seq2seq_attention_detailed(
    predictor: Any,
    image: Image.Image,
    settings: Settings,
) -> _DecoderEvidenceTraceOutcome:
    outcomes = _trace_seq2seq_attention_batch_detailed(
        predictor, [image], settings
    )
    return outcomes[0]


def _trace_seq2seq_attention(
    predictor: Any,
    image: Image.Image,
    settings: Settings,
) -> _DecoderEvidenceTrace | None:
    """Backward-compatible wrapper used by focused unit tests."""
    return _trace_seq2seq_attention_detailed(predictor, image, settings).trace


def _adaptive_decoder_thresholds(
    probabilities: np.ndarray,
    supports: np.ndarray,
    count: int,
    settings: Settings,
) -> tuple[float, float, float, float]:
    """Derive evidence thresholds relative to the stable part of one segment."""
    window = min(settings.decoder_evidence_window_tokens, count)
    baseline_end = max(
        settings.decoder_evidence_min_prefix_chars, count - window
    )
    baseline_probabilities = probabilities[:baseline_end]
    baseline_supports = supports[:baseline_end]
    probability_median = float(np.median(baseline_probabilities))
    support_median = float(np.median(baseline_supports))
    probability_threshold = min(
        settings.decoder_evidence_probability_threshold_cap,
        max(
            settings.decoder_evidence_low_token_probability,
            probability_median
            * settings.decoder_evidence_relative_probability_ratio,
        ),
    )
    support_threshold = min(
        settings.decoder_evidence_ink_threshold_cap,
        max(
            settings.decoder_evidence_low_ink_support,
            support_median * settings.decoder_evidence_relative_ink_ratio,
        ),
    )
    return (
        probability_threshold,
        support_threshold,
        probability_median,
        support_median,
    )


def _previous_lexical_word(text: str, before: int) -> re.Match[str] | None:
    matches = list(_TOKEN.finditer(text, 0, before))
    if not matches:
        return None
    match = matches[-1]
    if any(char.isalnum() or char == "_" for char in text[match.end() : before]):
        return None
    return match


def _expand_trim_start_to_attention_cluster(
    text: str,
    trim_start: int,
    count: int,
    probabilities: np.ndarray,
    centers: np.ndarray,
    supports: np.ndarray,
    trace: _DecoderEvidenceTrace,
    probability_threshold: float,
    support_threshold: float,
    settings: Settings,
) -> tuple[int, int]:
    """Expand a suspicious suffix to preceding words in the same visual cluster.

    Expansion is entirely evidence-based: the preceding word must share the
    stalled attention region and have weak pixel support relative to the stable
    prefix. Numeric words are never absorbed automatically.
    """
    if not settings.decoder_evidence_cluster_word_expansion_enabled:
        return trim_start, 0
    current = trim_start
    expanded = 0
    while expanded < settings.decoder_evidence_cluster_max_words:
        match = _previous_lexical_word(text, current)
        if match is None or any(char.isdigit() for char in match.group(0)):
            break
        if match.start() < settings.decoder_evidence_min_prefix_chars:
            break
        word_slice = slice(match.start(), match.end())
        combined_slice = slice(match.start(), count)
        stable_probabilities = probabilities[: match.start()]
        stable_supports = supports[: match.start()]
        if stable_probabilities.size < settings.decoder_evidence_min_prefix_chars:
            break

        word_probability = float(np.mean(probabilities[word_slice]))
        word_support = float(np.mean(supports[word_slice]))
        word_center = float(np.mean(centers[word_slice]))
        tail_center = float(np.mean(centers[current:count]))
        combined_range = float(np.ptp(centers[combined_slice]))
        stable_probability = float(np.median(stable_probabilities))
        stable_support = float(np.median(stable_supports))
        support_limit = min(
            settings.decoder_evidence_cluster_support_cap,
            max(
                support_threshold,
                stable_support * settings.decoder_evidence_cluster_support_ratio,
            ),
        )
        probability_limit = min(
            settings.decoder_evidence_cluster_probability_cap,
            max(
                probability_threshold
                + settings.decoder_evidence_cluster_probability_bonus,
                stable_probability
                * settings.decoder_evidence_cluster_probability_ratio,
            ),
        )
        same_cluster = (
            abs(word_center - tail_center)
            <= settings.decoder_evidence_cluster_center_distance_ratio
            and combined_range
            <= settings.decoder_evidence_cluster_attention_range_ratio
        )
        near_visual_end = (
            word_center
            >= trace.last_ink_position
            - settings.decoder_evidence_cluster_end_margin_ratio
        )
        visually_weak = word_support <= support_limit
        probability_compatible = (
            word_probability <= probability_limit
            or word_support <= support_threshold * 0.55
        )
        if not (same_cluster and near_visual_end and visually_weak and probability_compatible):
            break
        current = match.start()
        expanded += 1
    return current, expanded

def _word_evidence(
    text: str,
    probabilities: np.ndarray,
    centers: np.ndarray,
    supports: np.ndarray,
) -> list[dict[str, float | int | str]]:
    words: list[dict[str, float | int | str]] = []
    for match in _TOKEN.finditer(text):
        start, end = match.span()
        if end <= start:
            continue
        words.append(
            {
                "text": match.group(0),
                "start": start,
                "end": end,
                "probability": float(np.mean(probabilities[start:end])),
                "support": float(np.mean(supports[start:end])),
                "center": float(np.mean(centers[start:end])),
                "attention_range": float(np.ptp(centers[start:end])),
                "attention_progress": float(centers[end - 1] - centers[start]),
            }
        )
    return words


def _mean_word_metric(
    words: Sequence[dict[str, float | int | str]],
    key: str,
) -> float:
    return float(np.mean([float(word[key]) for word in words])) if words else 0.0


def _remove_evidence_span(text: str, start: int, end: int) -> str:
    """Remove one token span while preserving the surrounding text order.

    This function deliberately performs no spelling, grammar, punctuation or
    vocabulary correction. It only reconnects the untouched left/right spans.
    """
    left_raw = text[:start]
    right_raw = text[end:]
    had_boundary_space = bool(
        (left_raw and left_raw[-1].isspace())
        or (right_raw and right_raw[0].isspace())
    )
    left = left_raw.rstrip()
    right = right_raw.lstrip()
    if not left:
        return normalize_vietnamese_text(right)
    if not right:
        return normalize_vietnamese_text(left)
    needs_space = had_boundary_space or (
        (left[-1].isalnum() or left[-1] in ")]}")
        and (right[0].isalnum() or right[0] in "([{")
    )
    return normalize_vietnamese_text(left + (" " if needs_space else "") + right)


def _midline_unsupported_span_decision(
    trace: _DecoderEvidenceTrace,
    text: str,
    count: int,
    probabilities: np.ndarray,
    centers: np.ndarray,
    supports: np.ndarray,
    probability_threshold: float,
    support_threshold: float,
    settings: Settings,
) -> _DecoderEvidenceDecision | None:
    """Find an image-unsupported word span between two supported anchors.

    Lexical identity is never consulted. A candidate span must be weaker than
    both neighboring anchors, show stalled/recurrent attention, and be followed
    by a visually supported forward recovery. This makes the rule invariant to
    the actual words used by a document.
    """
    if not settings.decoder_evidence_midline_enabled:
        return None
    words = _word_evidence(text[:count], probabilities, centers, supports)
    if len(words) < 3:
        return None

    probability_limit = min(
        settings.decoder_evidence_probability_threshold_cap,
        probability_threshold + settings.decoder_evidence_midline_probability_bonus,
    )
    support_limit = min(
        settings.decoder_evidence_ink_threshold_cap,
        support_threshold * settings.decoder_evidence_midline_support_multiplier,
    )
    weak = [
        float(word["probability"]) <= probability_limit
        and float(word["support"]) <= support_limit
        for word in words
    ]

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, is_weak in enumerate(weak + [False]):
        if is_weak and run_start is None:
            run_start = index
            continue
        if is_weak or run_start is None:
            continue
        runs.append((run_start, index))
        run_start = None

    best: tuple[float, _DecoderEvidenceDecision] | None = None
    anchor_words = settings.decoder_evidence_midline_anchor_words
    for first, last in runs:
        word_count = last - first
        if (
            first <= 0
            or last >= len(words)
            or word_count < settings.decoder_evidence_midline_min_words
            or word_count > settings.decoder_evidence_midline_max_words
        ):
            continue
        span_start = int(words[first]["start"])
        span_end = int(words[last - 1]["end"])
        if (
            span_start < settings.decoder_evidence_min_prefix_chars
            or count - span_end < settings.decoder_evidence_midline_min_suffix_chars
            or (span_end - span_start) / max(1, count)
            > settings.decoder_evidence_midline_max_span_ratio
        ):
            continue
        removed = text[span_start:span_end].strip()
        if not removed:
            continue

        left_words = words[max(0, first - anchor_words) : first]
        right_words = words[last : min(len(words), last + anchor_words)]
        if not left_words or not right_words:
            continue
        span_words = words[first:last]

        span_probability = _mean_word_metric(span_words, "probability")
        span_support = _mean_word_metric(span_words, "support")
        span_center = _mean_word_metric(span_words, "center")
        left_probability = _mean_word_metric(left_words, "probability")
        left_support = _mean_word_metric(left_words, "support")
        left_center = _mean_word_metric(left_words, "center")
        right_probability = _mean_word_metric(right_words, "probability")
        right_support = _mean_word_metric(right_words, "support")
        right_center = _mean_word_metric(right_words, "center")

        span_centers = centers[span_start:span_end]
        span_range = float(np.ptp(span_centers))
        span_progress = float(span_centers[-1] - span_centers[0])
        anchor_progress = right_center - left_center
        right_recovery = right_center - span_center
        recurrent_distance = min(
            abs(span_center - left_center), abs(span_center - right_center)
        )

        anchors_supported = (
            left_probability
            >= max(
                probability_threshold,
                span_probability
                * settings.decoder_evidence_midline_anchor_probability_ratio,
            )
            and right_probability
            >= max(
                probability_threshold,
                span_probability
                * settings.decoder_evidence_midline_anchor_probability_ratio,
            )
            and left_support
            >= max(
                support_threshold,
                span_support * settings.decoder_evidence_midline_anchor_support_ratio,
            )
            and right_support
            >= max(
                support_threshold,
                span_support * settings.decoder_evidence_midline_anchor_support_ratio,
            )
        )
        stalled_or_recurrent = (
            span_range <= settings.decoder_evidence_midline_attention_range_ratio
            or abs(span_progress)
            <= settings.decoder_evidence_midline_attention_progress_ratio
        ) and (
            recurrent_distance
            <= settings.decoder_evidence_midline_recurrence_distance_ratio
            or right_recovery
            >= settings.decoder_evidence_midline_recovery_ratio
        )
        forward_recovery = (
            anchor_progress
            >= settings.decoder_evidence_midline_anchor_progress_ratio
            and right_center
            >= span_center - settings.decoder_evidence_midline_recurrence_distance_ratio
        )
        visually_isolated = (
            span_support
            <= min(left_support, right_support)
            * settings.decoder_evidence_midline_relative_support_ratio
            and span_probability
            <= min(left_probability, right_probability)
            * settings.decoder_evidence_midline_relative_probability_ratio
        )
        if not (
            anchors_supported
            and stalled_or_recurrent
            and forward_recovery
            and visually_isolated
        ):
            continue

        has_numeric = any(char.isdigit() for char in removed)
        validated = _remove_evidence_span(text, span_start, span_end)
        if len(validated) < settings.decoder_evidence_min_prefix_chars:
            continue
        score = (
            (min(left_support, right_support) - span_support)
            + (min(left_probability, right_probability) - span_probability)
            + max(0.0, anchor_progress)
            + max(0.0, right_recovery)
            - span_range
        )
        decision = _DecoderEvidenceDecision(
            text=validated,
            removed_text=removed,
            span_start=span_start,
            span_end=span_end,
            reason="unsupported_midline_span",
            span_kind="midline",
            numeric_only_warning=has_numeric,
            probability_threshold=probability_threshold,
            ink_support_threshold=support_threshold,
            span_probability_mean=span_probability,
            span_ink_support_mean=span_support,
            span_attention_range=span_range,
            span_attention_progress=span_progress,
            left_anchor_probability_mean=left_probability,
            left_anchor_ink_support_mean=left_support,
            left_anchor_attention_mean=left_center,
            right_anchor_probability_mean=right_probability,
            right_anchor_ink_support_mean=right_support,
            right_anchor_attention_mean=right_center,
        )
        if best is None or score > best[0]:
            best = (score, decision)
    return best[1] if best is not None else None


def _decoder_evidence_decision(
    trace: _DecoderEvidenceTrace,
    settings: Settings,
) -> _DecoderEvidenceDecision | None:
    text = trace.raw_text
    count = min(
        len(text),
        len(trace.token_probabilities),
        len(trace.attention_centers),
        len(trace.ink_support),
    )
    if count < settings.decoder_evidence_min_prefix_chars + settings.decoder_evidence_min_unsupported_tokens:
        return None
    probabilities = np.asarray(trace.token_probabilities[:count], dtype=np.float32)
    centers = np.asarray(trace.attention_centers[:count], dtype=np.float32)
    supports = np.asarray(trace.ink_support[:count], dtype=np.float32)
    (
        probability_threshold,
        support_threshold,
        _baseline_probability,
        _baseline_support,
    ) = _adaptive_decoder_thresholds(
        probabilities, supports, count, settings
    )
    window = min(settings.decoder_evidence_window_tokens, count)
    recent_prob = probabilities[-window:]
    recent_center = centers[-window:]
    recent_support = supports[-window:]
    stalled = (
        float(np.ptp(recent_center)) <= settings.decoder_evidence_stall_range_ratio
        and float(recent_prob.mean()) <= probability_threshold + 0.05
        and float(recent_support.mean()) <= support_threshold + 0.03
    )
    exhausted = (
        float(recent_center.mean())
        >= trace.last_ink_position - settings.decoder_evidence_end_margin_ratio
        and float(recent_support.mean()) <= support_threshold
        and float(recent_prob.mean()) <= probability_threshold + 0.04
    )
    near_loop_start = _near_repetitive_suffix_start(
        text[:count],
        settings.decoder_evidence_near_loop_similarity,
        settings.decoder_evidence_near_loop_min_tokens,
    )

    unsupported = (
        (probabilities < probability_threshold)
        & (supports < support_threshold)
    )
    run_start = count
    while run_start > 0 and bool(unsupported[run_start - 1]):
        run_start -= 1
    unsupported_count = count - run_start

    tail_stalled = False
    tail_exhausted = False
    if unsupported_count >= settings.decoder_evidence_min_unsupported_tokens:
        tail_centers = centers[run_start:count]
        tail_support = supports[run_start:count]
        tail_prob = probabilities[run_start:count]
        tail_stalled = (
            float(np.ptp(tail_centers)) <= settings.decoder_evidence_stall_range_ratio
            and float(tail_prob.mean()) <= probability_threshold
            and float(tail_support.mean()) <= support_threshold
        )
        tail_exhausted = (
            float(tail_centers.mean())
            >= trace.last_ink_position - settings.decoder_evidence_end_margin_ratio
            and float(tail_support.mean()) <= support_threshold
            and float(tail_prob.mean()) <= probability_threshold
        )

    reason: str | None = None
    trim_start: int | None = None
    if near_loop_start is not None and (stalled or exhausted or tail_stalled or tail_exhausted):
        trim_start = near_loop_start
        reason = "attention_near_loop"
    elif unsupported_count >= settings.decoder_evidence_min_unsupported_tokens and tail_exhausted:
        trim_start = run_start
        reason = "visual_evidence_exhausted"
    elif unsupported_count >= settings.decoder_evidence_min_unsupported_tokens and tail_stalled:
        trim_start = run_start
        reason = "attention_stall"

    if trim_start is not None:
        expanded_word_count = 0
        if reason in {"visual_evidence_exhausted", "attention_stall"}:
            trim_start, expanded_word_count = _expand_trim_start_to_attention_cluster(
                text,
                trim_start,
                count,
                probabilities,
                centers,
                supports,
                trace,
                probability_threshold,
                support_threshold,
                settings,
            )

        while trim_start > 0 and text[trim_start - 1].isspace():
            trim_start -= 1
        prefix = text[:trim_start].rstrip()
        removed = text[trim_start:count].strip()
        if len(prefix) >= settings.decoder_evidence_min_prefix_chars and removed:
            has_numeric = any(char.isdigit() for char in removed)
            span_slice = slice(trim_start, count)
            return _DecoderEvidenceDecision(
                text=normalize_vietnamese_text(prefix),
                removed_text=removed,
                span_start=trim_start,
                span_end=count,
                reason=str(reason),
                span_kind="suffix",
                numeric_only_warning=has_numeric,
                expanded_word_count=expanded_word_count,
                probability_threshold=probability_threshold,
                ink_support_threshold=support_threshold,
                span_probability_mean=float(np.mean(probabilities[span_slice])),
                span_ink_support_mean=float(np.mean(supports[span_slice])),
                span_attention_range=float(np.ptp(centers[span_slice])),
                span_attention_progress=float(
                    centers[count - 1] - centers[trim_start]
                ),
            )

    return _midline_unsupported_span_decision(
        trace,
        text,
        count,
        probabilities,
        centers,
        supports,
        probability_threshold,
        support_threshold,
        settings,
    )

def _global_attention_centers(
    trace: _DecoderEvidenceTrace,
    segment: _Segment,
    original_line_width: int,
    inference_image_width: int | None = None,
) -> np.ndarray:
    """Map local attention to the coordinate system of the original line crop."""
    if not trace.attention_centers:
        return np.zeros(0, dtype=np.float32)
    local = np.asarray(trace.attention_centers, dtype=np.float32)
    source_width = max(1, segment.source_right - segment.source_left)
    image_width = max(1, inference_image_width or segment.image.width)
    border = 2 if segment.was_split else 0
    source_x = np.clip(local * image_width - border, 0.0, float(source_width))
    return np.clip(
        (float(segment.source_left) + source_x) / max(1.0, float(original_line_width)),
        0.0,
        1.0,
    ).astype(np.float32)


def _word_evidence_extended(trace: _DecoderEvidenceTrace) -> list[dict[str, float | int | str]]:
    count = min(
        len(trace.raw_text), len(trace.token_probabilities),
        len(trace.attention_centers), len(trace.ink_support),
    )
    probabilities = np.asarray(trace.token_probabilities[:count], dtype=np.float32)
    centers = np.asarray(trace.attention_centers[:count], dtype=np.float32)
    supports = np.asarray(trace.ink_support[:count], dtype=np.float32)
    coverage = _trace_metric_array(trace, "visual_coverage_gain", count, 1.0)
    reuse = _trace_metric_array(trace, "reused_attention_ratio", count, 0.0)
    words = _word_evidence(trace.raw_text[:count], probabilities, centers, supports)
    for word in words:
        start, end = int(word["start"]), int(word["end"])
        word["coverage_gain"] = float(np.mean(coverage[start:end]))
        word["reuse_ratio"] = float(np.mean(reuse[start:end]))
    return words


def _cross_segment_suffix_decision(
    left_trace: _DecoderEvidenceTrace,
    right_trace: _DecoderEvidenceTrace,
    left_segment: _Segment,
    right_segment: _Segment,
    original_line_width: int,
    left_inference_width: int,
    right_inference_width: int,
    settings: Settings,
) -> _DecoderEvidenceDecision | None:
    """Use the next split as a right visual anchor for a weak tail."""
    if not settings.decoder_evidence_cross_segment_enabled:
        return None
    if not left_trace.visual_coverage_gain or not right_trace.visual_coverage_gain:
        return None
    left_words = _word_evidence_extended(left_trace)
    right_words = _word_evidence_extended(right_trace)
    if len(left_words) < 2 or not right_words:
        return None
    anchor_n = min(settings.decoder_evidence_cross_segment_anchor_words, len(right_words))
    right_anchor = right_words[:anchor_n]
    right_prob = _mean_word_metric(right_anchor, "probability")
    right_support = _mean_word_metric(right_anchor, "support")
    right_coverage = _mean_word_metric(right_anchor, "coverage_gain")
    left_count = min(
        len(left_trace.raw_text), len(left_trace.token_probabilities),
        len(left_trace.attention_centers), len(left_trace.ink_support),
    )
    probabilities = np.asarray(left_trace.token_probabilities[:left_count], dtype=np.float32)
    supports = np.asarray(left_trace.ink_support[:left_count], dtype=np.float32)
    prob_threshold, support_threshold, _, _ = _adaptive_decoder_thresholds(
        probabilities, supports, left_count, settings
    )
    global_left = _global_attention_centers(
        left_trace, left_segment, original_line_width, left_inference_width
    )
    global_right = _global_attention_centers(
        right_trace, right_segment, original_line_width, right_inference_width
    )
    right_chars = int(right_anchor[-1]["end"])
    right_global_mean = float(np.mean(global_right[:right_chars])) if right_chars else 0.0
    max_words = min(settings.decoder_evidence_cross_segment_max_words, len(left_words) - 1)
    best: tuple[int, float, _DecoderEvidenceDecision] | None = None
    for word_count in range(1, max_words + 1):
        first = len(left_words) - word_count
        span_words = left_words[first:]
        left_anchor = left_words[
            max(0, first - settings.decoder_evidence_cross_segment_anchor_words):first
        ]
        if not left_anchor:
            continue
        span_start = int(span_words[0]["start"])
        span_end = int(span_words[-1]["end"])
        if span_start < settings.decoder_evidence_min_prefix_chars:
            continue
        removed = left_trace.raw_text[span_start:span_end].strip()
        if not removed:
            continue
        span_prob = _mean_word_metric(span_words, "probability")
        span_support = _mean_word_metric(span_words, "support")
        span_coverage = _mean_word_metric(span_words, "coverage_gain")
        span_reuse = _mean_word_metric(span_words, "reuse_ratio")
        left_prob = _mean_word_metric(left_anchor, "probability")
        left_support = _mean_word_metric(left_anchor, "support")
        left_coverage = _mean_word_metric(left_anchor, "coverage_gain")
        visual_ref = max(1e-6, min(left_coverage, right_coverage))
        support_ref = max(1e-6, min(left_support, right_support))
        prob_ref = max(1e-6, min(left_prob, right_prob))
        span_global = global_left[span_start:span_end]
        left_start, left_end = int(left_anchor[0]["start"]), int(left_anchor[-1]["end"])
        left_global_mean = float(np.mean(global_left[left_start:left_end]))
        span_global_mean = float(np.mean(span_global)) if span_global.size else left_global_mean
        span_progress = float(span_global[-1] - span_global[0]) if span_global.size else 0.0
        if right_prob < max(prob_threshold, settings.decoder_evidence_cross_segment_min_anchor_probability):
            continue
        if right_support < max(support_threshold, settings.decoder_evidence_cross_segment_min_anchor_support):
            continue
        support_limit = support_ref * settings.decoder_evidence_cross_segment_relative_support_ratio
        coverage_limit = max(
            settings.decoder_evidence_cross_segment_coverage_gain_cap,
            visual_ref * settings.decoder_evidence_cross_segment_relative_coverage_ratio,
        )
        probability_limit = prob_ref * settings.decoder_evidence_cross_segment_relative_probability_ratio
        per_word_weak = all(
            float(word["support"]) <= support_limit
            and float(word["coverage_gain"]) <= coverage_limit
            and (
                float(word["probability"]) <= probability_limit
                or float(word["support"])
                <= settings.decoder_evidence_cross_segment_strong_visual_override_support
            )
            for word in span_words
        )
        if not per_word_weak:
            continue
        if span_support > support_limit or span_coverage > coverage_limit:
            continue
        if not (
            span_prob <= probability_limit
            or span_support <= settings.decoder_evidence_cross_segment_strong_visual_override_support
        ):
            continue
        if not (
            span_reuse >= settings.decoder_evidence_cross_segment_min_reuse_ratio
            or abs(span_progress) <= settings.decoder_evidence_cross_segment_max_progress_ratio
        ):
            continue
        if right_global_mean < span_global_mean - settings.decoder_evidence_cross_segment_global_backtrack_tolerance:
            continue
        prefix = normalize_vietnamese_text(left_trace.raw_text[:span_start].rstrip())
        if len(prefix) < settings.decoder_evidence_min_prefix_chars:
            continue
        has_numeric = any(char.isdigit() for char in removed)
        score = (support_ref - span_support) + (visual_ref - span_coverage) + span_reuse
        decision = _DecoderEvidenceDecision(
            text=prefix, removed_text=removed, span_start=span_start, span_end=span_end,
            reason="cross_segment_visual_gap", span_kind="cross_segment",
            numeric_only_warning=has_numeric,
            probability_threshold=prob_threshold, ink_support_threshold=support_threshold,
            span_probability_mean=span_prob, span_ink_support_mean=span_support,
            span_attention_range=float(np.ptp(span_global)) if span_global.size else 0.0,
            span_attention_progress=span_progress,
            left_anchor_probability_mean=left_prob, left_anchor_ink_support_mean=left_support,
            left_anchor_attention_mean=left_global_mean,
            right_anchor_probability_mean=right_prob, right_anchor_ink_support_mean=right_support,
            right_anchor_attention_mean=right_global_mean,
            span_visual_coverage_gain_mean=span_coverage,
            span_reused_attention_ratio_mean=span_reuse,
            left_anchor_visual_coverage_gain_mean=left_coverage,
            right_anchor_visual_coverage_gain_mean=right_coverage,
            global_span_start=float(span_global[0]) if span_global.size else 0.0,
            global_span_end=float(span_global[-1]) if span_global.size else 0.0,
            right_anchor_global_attention_mean=right_global_mean,
        )
        # Prefer the longest tail among similarly confident candidates.
        candidate = (word_count, score, decision)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else None


def _cross_segment_rejection_reasons(
    left_trace: _DecoderEvidenceTrace,
    right_trace: _DecoderEvidenceTrace,
    settings: Settings,
) -> tuple[str, ...]:
    if not left_trace.visual_coverage_gain or not right_trace.visual_coverage_gain:
        return ("coverage_metrics_unavailable",)
    left_words = _word_evidence_extended(left_trace)
    right_words = _word_evidence_extended(right_trace)
    if len(left_words) < 2 or not right_words:
        return ("insufficient_words",)
    tail = left_words[-min(3, len(left_words)):]
    head = right_words[:min(2, len(right_words))]
    reasons: list[str] = []
    if _mean_word_metric(tail, "coverage_gain") >= _mean_word_metric(head, "coverage_gain") * settings.decoder_evidence_cross_segment_relative_coverage_ratio:
        reasons.append("tail_coverage_not_lower")
    if _mean_word_metric(tail, "support") >= _mean_word_metric(head, "support") * settings.decoder_evidence_cross_segment_relative_support_ratio:
        reasons.append("tail_support_not_lower")
    if _mean_word_metric(tail, "reuse_ratio") < settings.decoder_evidence_cross_segment_min_reuse_ratio:
        reasons.append("tail_attention_not_reused")
    return tuple(reasons or ["boundary_not_confident"])


def _cross_segment_rejection_diagnostics(
    left_trace: _DecoderEvidenceTrace,
    right_trace: _DecoderEvidenceTrace,
) -> dict[str, float]:
    """Return raw tail/head evidence so threshold failures are auditable."""
    left_words = _word_evidence_extended(left_trace)
    right_words = _word_evidence_extended(right_trace)
    tail = left_words[-min(3, len(left_words)):]
    head = right_words[:min(2, len(right_words))]
    return {
        "tail_probability_mean": round(_mean_word_metric(tail, "probability"), 4),
        "tail_support_mean": round(_mean_word_metric(tail, "support"), 4),
        "tail_coverage_gain_mean": round(_mean_word_metric(tail, "coverage_gain"), 4),
        "tail_reuse_ratio_mean": round(_mean_word_metric(tail, "reuse_ratio"), 4),
        "right_probability_mean": round(_mean_word_metric(head, "probability"), 4),
        "right_support_mean": round(_mean_word_metric(head, "support"), 4),
        "right_coverage_gain_mean": round(_mean_word_metric(head, "coverage_gain"), 4),
        "right_reuse_ratio_mean": round(_mean_word_metric(head, "reuse_ratio"), 4),
    }


def _split_wide_image(
    image: Image.Image,
    image_height: int,
    image_max_width: int,
    target_ratio: float,
    max_segments: int,
    valley_max_ink_ratio: float,
    overlap_height_ratio: float,
) -> list[_SplitImage]:
    """Chia crop width-cap tại valley và giữ overlap pixel quanh seam.

    Core segment được chọn đủ hẹp để sau khi mở rộng hai phía một khoảng overlap,
    ảnh vẫn không vượt ``image_max_width``. Overlap giúp giữ từ ngắn nằm sát điểm
    cắt; phần text trùng được loại ở bước seam-aware merge.
    """
    current_width = _resized_width(image, image_height)
    full_ink, full_ink_columns = _image_ink_stats(image)
    if current_width <= image_max_width:
        return [
            _SplitImage(
                image=image.convert("RGB"),
                core_left=0,
                core_right=image.width,
                source_left=0,
                source_right=image.width,
                left_overlap=0,
                right_overlap=0,
                ink_ratio=full_ink,
                ink_columns_ratio=full_ink_columns,
                is_tail=True,
            )
        ]

    desired_overlap = max(0, int(round(image.height * overlap_height_ratio)))
    safe_model_width = max(64, min(image_max_width, int(image_max_width * target_ratio)))
    target_source_width = max(
        24,
        math.floor(safe_model_width * image.height / max(image_height, 1)),
    )
    # Chừa headroom cho overlap và border trắng 2px mỗi phía.
    model_cap_source_width = max(
        target_source_width,
        math.floor((image_max_width - 10) * image.height / max(image_height, 1)) - 4,
    )
    core_cap_source_width = max(
        24,
        model_cap_source_width - 2 * desired_overlap,
    )
    target_source_width = min(target_source_width, core_cap_source_width)
    minimum_segment = max(20, int(target_source_width * 0.24))

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink_projection = (gray < 210).mean(axis=0).astype(np.float32)
    smooth = _smooth_projection(ink_projection, max(1, image.height // 8))
    eligible = smooth <= valley_max_ink_ratio

    candidates: list[tuple[int, int, float]] = []
    run_start: int | None = None
    for index, is_gap in enumerate(eligible.tolist() + [False]):
        if is_gap and run_start is None:
            run_start = index
            continue
        if is_gap or run_start is None:
            continue
        run_end = index
        run_width = run_end - run_start
        center = (run_start + run_end) // 2
        if run_width >= 2 and minimum_segment <= center <= image.width - minimum_segment:
            candidates.append((center, run_width, float(smooth[center])))
        run_start = None

    if not candidates:
        return [
            _SplitImage(
                image=image.convert("RGB"),
                core_left=0,
                core_right=image.width,
                source_left=0,
                source_right=image.width,
                left_overlap=0,
                right_overlap=0,
                ink_ratio=full_ink,
                ink_columns_ratio=full_ink_columns,
                is_tail=True,
            )
        ]

    cuts = [0]
    start_x = 0
    while image.width - start_x > core_cap_source_width and len(cuts) < max_segments:
        desired = start_x + target_source_width
        hard_limit = min(image.width - minimum_segment, start_x + core_cap_source_width)
        lower = start_x + minimum_segment
        available = [
            item
            for item in candidates
            if lower <= item[0] <= hard_limit and item[0] > cuts[-1]
        ]
        if not available:
            return [
                _SplitImage(
                    image=image.convert("RGB"),
                    core_left=0,
                    core_right=image.width,
                    source_left=0,
                    source_right=image.width,
                    left_overlap=0,
                    right_overlap=0,
                    ink_ratio=full_ink,
                    ink_columns_ratio=full_ink_columns,
                    is_tail=True,
                )
            ]
        cut, _, _ = min(
            available,
            key=lambda item: (abs(item[0] - desired), -item[1], item[2]),
        )
        cuts.append(cut)
        start_x = cut

    cuts.append(image.width)
    if len(cuts) <= 2:
        return [
            _SplitImage(
                image=image.convert("RGB"),
                core_left=0,
                core_right=image.width,
                source_left=0,
                source_right=image.width,
                left_overlap=0,
                right_overlap=0,
                ink_ratio=full_ink,
                ink_columns_ratio=full_ink_columns,
                is_tail=True,
            )
        ]

    split_images: list[_SplitImage] = []
    core_pairs = list(zip(cuts, cuts[1:]))
    for segment_index, (core_left, core_right) in enumerate(core_pairs):
        if core_right - core_left < minimum_segment:
            return [
                _SplitImage(
                    image=image.convert("RGB"),
                    core_left=0,
                    core_right=image.width,
                    source_left=0,
                    source_right=image.width,
                    left_overlap=0,
                    right_overlap=0,
                    ink_ratio=full_ink,
                    ink_columns_ratio=full_ink_columns,
                    is_tail=True,
                )
            ]

        left_overlap = min(desired_overlap, core_left)
        right_overlap = min(desired_overlap, image.width - core_right)
        source_left = core_left - left_overlap
        source_right = core_right + right_overlap

        # Nếu overlap khiến vượt cap, giảm cân bằng hai phía nhưng không đụng core.
        overflow = max(0, source_right - source_left - model_cap_source_width)
        while overflow > 0 and (left_overlap > 0 or right_overlap > 0):
            if right_overlap >= left_overlap and right_overlap > 0:
                right_overlap -= 1
                source_right -= 1
            elif left_overlap > 0:
                left_overlap -= 1
                source_left += 1
            overflow -= 1

        segment = image.crop((source_left, 0, source_right, image.height)).convert("RGB")
        segment = ImageOps.expand(segment, border=(2, 0, 2, 0), fill="white")
        if _resized_width(segment, image_height) > image_max_width:
            return [
                _SplitImage(
                    image=image.convert("RGB"),
                    core_left=0,
                    core_right=image.width,
                    source_left=0,
                    source_right=image.width,
                    left_overlap=0,
                    right_overlap=0,
                    ink_ratio=full_ink,
                    ink_columns_ratio=full_ink_columns,
                    is_tail=True,
                )
            ]
        ink_ratio, ink_columns_ratio = _image_ink_stats(segment)
        split_images.append(
            _SplitImage(
                image=segment,
                core_left=core_left,
                core_right=core_right,
                source_left=source_left,
                source_right=source_right,
                left_overlap=left_overlap,
                right_overlap=right_overlap,
                ink_ratio=ink_ratio,
                ink_columns_ratio=ink_columns_ratio,
                is_tail=segment_index == len(core_pairs) - 1,
            )
        )
    return split_images if len(split_images) > 1 else split_images

def _pad_to_resized_width(
    image: Image.Image,
    target_resized_width: int,
    image_height: int,
) -> tuple[Image.Image, int]:
    """Pad trắng bên phải để VietOCR gom các ảnh vào cùng exact-width bucket."""
    current = _resized_width(image, image_height)
    if current >= target_resized_width:
        return image, 0
    target_source_width = max(
        image.width,
        math.floor(target_resized_width * image.height / image_height),
    )
    if target_source_width <= image.width:
        return image, 0
    canvas = Image.new("RGB", (target_source_width, image.height), "white")
    canvas.paste(image.convert("RGB"), (0, 0))
    return canvas, target_source_width - image.width

def _find_decoder_loop(text: str) -> _LoopIssue | None:
    """Phát hiện loop n-gram hoàn chỉnh và ``2 lần + prefix lần 3`` ở cuối.

    Hai lần lặp thuần túy vẫn không bị cắt để tránh false-positive. Trường hợp
    partial chỉ được chấp nhận với n-gram từ hai token, có hai lần đầy đủ và phần
    còn lại ở cuối chuỗi trùng ít nhất nửa prefix của lần thứ ba.
    """
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    if len(tokens) < 3:
        return None
    best: _LoopIssue | None = None

    for ngram_size in range(1, min(4, len(tokens) // 3) + 1):
        minimum_tokens = ngram_size * 3
        if len(tokens) < minimum_tokens:
            continue
        for start in range(0, len(tokens) - minimum_tokens + 1):
            unit = tokens[start : start + ngram_size]
            repeats = 1
            cursor = start + ngram_size
            while cursor + ngram_size <= len(tokens):
                if tokens[cursor : cursor + ngram_size] != unit:
                    break
                repeats += 1
                cursor += ngram_size
            if repeats < 3:
                continue
            issue = _LoopIssue(start, ngram_size, repeats)
            if best is None or issue.severity > best.severity:
                best = issue

    # Bắt loop dạng: [A B C D] [A B C D] [A B ...] ở cuối output.
    for ngram_size in range(2, min(4, len(tokens) // 2) + 1):
        for start in range(0, len(tokens) - ngram_size * 2):
            unit = tokens[start : start + ngram_size]
            second = tokens[start + ngram_size : start + ngram_size * 2]
            if second != unit:
                continue
            cursor = start + ngram_size * 2
            remaining = tokens[cursor:]
            if not remaining or len(remaining) >= ngram_size:
                continue
            minimum_partial = max(1, math.ceil(ngram_size * 0.5))
            if len(remaining) < minimum_partial or remaining != unit[: len(remaining)]:
                continue
            issue = _LoopIssue(start, ngram_size, 2, partial_tokens=len(remaining))
            if best is None or issue.severity > best.severity:
                best = issue
    return best

def _decoder_loop_removed_contains_numeric(
    text: str,
    issue: _LoopIssue | None,
) -> bool:
    if issue is None:
        return False
    matches = list(_TOKEN.finditer(text))
    if issue.start_token >= len(matches):
        return False
    start_char = matches[issue.start_token].start()
    return any(character.isdigit() for character in text[start_char:])


def _trim_decoder_loop(text: str, issue: _LoopIssue | None) -> str:
    if issue is None:
        return text
    matches = list(_TOKEN.finditer(text))
    if issue.start_token >= len(matches):
        return text
    start_char = matches[issue.start_token].start()
    if _decoder_loop_removed_contains_numeric(text, issue):
        return text
    trimmed = text[:start_char].rstrip(" ,;:-")
    # Loop ở đầu dòng hoặc chỉ có một token rác trước loop: bỏ cả dòng. Giữ lại
    # prefix từ hai token trở lên vì đó thường là phần câu hợp lệ trước hallucination.
    if len(_TOKEN.findall(trimmed)) < 2:
        return ""
    return trimmed if len(trimmed) >= 4 else ""


def _find_character_loop(text: str) -> int | None:
    """Trả vị trí bắt đầu chuỗi ký tự/dấu câu lặp bất thường."""
    starts = [
        match.start()
        for pattern in (_CHAR_RUN, _PUNCT_RUN)
        if (match := pattern.search(text)) is not None
    ]
    return min(starts) if starts else None


def _trim_character_loop(text: str, start_char: int | None) -> str:
    if start_char is None:
        return text
    if any(character.isdigit() for character in text[start_char:]):
        return text
    trimmed = text[:start_char].rstrip(" ,.;:_-")
    # Một prefix quá ngắn thường chỉ là nhiễu ở con dấu/lề trang.
    if len(_TOKEN.findall(trimmed)) < 2:
        return ""
    return trimmed


def _numeric_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _TOKEN.findall(text)
        if any(character.isdigit() for character in token)
    )


def _inference_context(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    if not enabled:
        return contextlib.nullcontext()
    try:
        import torch

        return torch.inference_mode()
    except Exception:
        return contextlib.nullcontext()


def _unpack_batch(value: Any) -> tuple[list[str], list[float]]:
    if isinstance(value, tuple) and len(value) == 2:
        texts, probabilities = value
        return list(texts), [float(probability) for probability in probabilities]
    if isinstance(value, list) and all(
        isinstance(item, tuple) and len(item) == 2 for item in value
    ):
        return [str(item[0]) for item in value], [float(item[1]) for item in value]
    raise ValueError("VietOCR trả kết quả batch không đúng định dạng")


def _adaptive_batches(
    segments: Sequence[_Segment],
    max_batch_size: int,
    pixel_budget: int,
    width_ratio: float,
) -> Iterator[list[int]]:
    ordered = sorted(range(len(segments)), key=lambda index: segments[index].resized_width)
    batch: list[int] = []
    min_width = max_width = 0
    for index in ordered:
        width = segments[index].resized_width
        prospective_min = width if not batch else min(min_width, width)
        prospective_max = width if not batch else max(max_width, width)
        prospective_count = len(batch) + 1
        exceeds_size = prospective_count > max_batch_size
        exceeds_budget = prospective_max * prospective_count > pixel_budget
        exceeds_ratio = (
            bool(batch)
            and prospective_max / max(prospective_min, 1) > width_ratio
        )
        if batch and (exceeds_size or exceeds_budget or exceeds_ratio):
            yield batch
            batch = []
            prospective_min = prospective_max = width
        batch.append(index)
        min_width, max_width = prospective_min, prospective_max
    if batch:
        yield batch


def _remove_leading_tokens(text: str, count: int) -> str:
    matches = list(_TOKEN.finditer(text))
    if count <= 0 or not matches:
        return text
    if count >= len(matches):
        return text[matches[-1].end() :].lstrip(" ,;:-")
    return text[matches[count - 1].end() :].lstrip(" ,;:-")


def _merge_text_pair(
    left: str,
    right: str,
    max_token_overlap: int,
    max_char_overlap: int,
) -> tuple[str, bool]:
    left = normalize_vietnamese_text(left)
    right = normalize_vietnamese_text(right)
    if not left:
        return right, False
    if not right:
        return left, False

    left_tokens = [match.group(0).casefold() for match in _TOKEN.finditer(left)]
    right_tokens = [match.group(0).casefold() for match in _TOKEN.finditer(right)]
    maximum = min(max_token_overlap, len(left_tokens), len(right_tokens))
    for size in range(maximum, 0, -1):
        if left_tokens[-size:] == right_tokens[:size]:
            tail = _remove_leading_tokens(right, size)
            if not tail:
                return left, True
            separator = "" if tail[:1] in ",.;:!?)]}" else " "
            return normalize_vietnamese_text(f"{left}{separator}{tail}"), True

    # Fallback ký tự cho overlap bị tokenizer tách khác nhau bởi dấu câu. Chỉ
    # chấp nhận overlap đủ dài và bắt đầu/kết thúc trên biên từ để tránh xóa nhầm.
    folded_left = left.casefold()
    folded_right = right.casefold()
    maximum_chars = min(max_char_overlap, len(left), len(right))
    for size in range(maximum_chars, 3, -1):
        if folded_left[-size:] != folded_right[:size]:
            continue
        left_boundary = len(left) == size or not left[-size - 1].isalnum()
        right_boundary = len(right) == size or not right[size].isalnum()
        if not (left_boundary or right_boundary):
            continue
        tail = right[size:].lstrip()
        separator = "" if not tail or tail[:1] in ",.;:!?)]}" else " "
        return normalize_vietnamese_text(f"{left}{separator}{tail}"), True

    separator = "" if right[:1] in ",.;:!?)]}" else " "
    return normalize_vietnamese_text(f"{left}{separator}{right}"), False


def _join_segments(
    texts: Sequence[str],
    max_token_overlap: int = 4,
    max_char_overlap: int = 32,
) -> tuple[str, int]:
    result = ""
    overlap_merge_count = 0
    for raw in texts:
        text = normalize_vietnamese_text(raw)
        if not text:
            continue
        result, merged = _merge_text_pair(
            result,
            text,
            max_token_overlap=max_token_overlap,
            max_char_overlap=max_char_overlap,
        )
        overlap_merge_count += int(merged)
    return normalize_vietnamese_text(result), overlap_merge_count


class VietOcrRecognizer:
    """VietOCR persistent với split crop dài, adaptive batching và retry có ngân sách."""

    def __init__(self, settings: Settings, predictor: Any | None = None) -> None:
        self.settings = settings
        self.predictor = predictor
        self._secondary_model: Any | None = None
        self._secondary_model_error: str | None = None
        self._secondary_mkldnn_effective: bool | None = None
        self._secondary_fallback_used = False
        self.last_batch_metrics: dict[str, Any] = {}
        self.last_page_metrics: dict[int, dict[str, Any]] = {}
        self._semantic_retry_cache: dict[str, tuple[RetryVariant, ...]] = {}

    def warm(self) -> None:
        self._predictor()

    def recognize_targeted(self, crops: Sequence[LineCrop]) -> list[Recognition]:
        """Nhận dạng lại một số crop nhỏ mà không ghi đè metrics batch chính."""
        if not crops:
            return []
        predictor = self._predictor()
        recognitions = [self._recognize_one(predictor, crop.image) for crop in crops]
        # Targeted retries happen after the main recognition pass. They must
        # cross the same semantic boundary before the pipeline may expose them.
        return self._apply_semantic_verification(crops, recognitions, {})

    def _predictor(self) -> Any:
        if self.predictor is not None:
            return self.predictor
        if importlib.util.find_spec("pkg_resources") is None:
            raise RuntimeError(
                "VietOCR 0.3.13 phụ thuộc gdown 4.4.0, cần pkg_resources. "
                "Hãy chạy: python -m pip install --force-reinstall setuptools==80.10.2"
            )
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        config_path = self.settings.resolve_project_path(self.settings.vietocr_config_path)
        weights_path = self.settings.resolve_project_path(self.settings.vietocr_weights_path)
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "Thiếu models/vietocr/vgg_seq2seq.yml hoặc vgg_seq2seq.pth; xem README.md"
            )
        config = Cfg.load_config_from_file(str(config_path))
        checkpoint_height = int(config.get("dataset", {}).get("image_height", 0))
        if checkpoint_height != self.settings.vietocr_image_height:
            raise ValueError(
                "Sai kích thước ảnh VietOCR: "
                f"checkpoint={checkpoint_height}, runtime={self.settings.vietocr_image_height}"
            )
        config["weights"] = str(weights_path)
        config.setdefault("cnn", {})["pretrained"] = False
        config["device"] = self.settings.vietocr_device
        config.setdefault("predictor", {})["beamsearch"] = self.settings.vietocr_beamsearch
        self.predictor = _build_predictor_with_safe_weights(
            Predictor, config, self.settings.vietocr_safe_weights_only
        )
        return self.predictor

    def _secondary_recognizer(self) -> Any | None:
        if not self.settings.secondary_recognizer_enabled:
            return None
        if self._secondary_model is not None:
            return self._secondary_model
        if self._secondary_model_error is not None:
            return None
        try:
            from paddleocr import TextRecognition

            self._secondary_model = TextRecognition(
                model_name=self.settings.secondary_recognizer_model_name,
                device=self.settings.paddle_device,
                enable_mkldnn=self.settings.paddle_enable_mkldnn,
                cpu_threads=self.settings.paddle_cpu_threads,
            )
            self._secondary_mkldnn_effective = self.settings.paddle_enable_mkldnn
            return self._secondary_model
        except Exception as exc:
            self._secondary_model_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            return None

    @staticmethod
    def _is_onednn_pir_error(exc: BaseException) -> bool:
        message = str(exc)
        return bool(
            isinstance(exc, NotImplementedError)
            and "ConvertPirAttribute2RuntimeAttribute" in message
            and "onednn" in message.casefold()
        )

    def _rebuild_secondary_without_mkldnn(self) -> Any | None:
        try:
            from paddleocr import TextRecognition

            self._secondary_model = TextRecognition(
                model_name=self.settings.secondary_recognizer_model_name,
                device=self.settings.paddle_device,
                enable_mkldnn=False,
                cpu_threads=self.settings.paddle_cpu_threads,
            )
            self._secondary_model_error = None
            self._secondary_mkldnn_effective = False
            self._secondary_fallback_used = True
            return self._secondary_model
        except Exception as exc:
            self._secondary_model = None
            self._secondary_model_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            return None

    @staticmethod
    def _parse_secondary_item(item: Any) -> _SecondaryRecognition | None:
        candidates: list[Any] = [item]
        json_value = getattr(item, "json", None)
        if callable(json_value):
            try:
                candidates.append(json_value())
            except Exception:
                pass
        elif json_value is not None:
            candidates.append(json_value)
        for candidate in candidates:
            if isinstance(candidate, dict):
                data = candidate.get("res", candidate)
                if isinstance(data, dict) and data.get("rec_text") is not None:
                    return _SecondaryRecognition(
                        text=normalize_vietnamese_text(str(data.get("rec_text", ""))),
                        confidence=float(data.get("rec_score", 0.0) or 0.0),
                    )
            try:
                text = candidate["rec_text"]
                score = candidate["rec_score"]
            except Exception:
                continue
            return _SecondaryRecognition(
                text=normalize_vietnamese_text(str(text)),
                confidence=float(score),
            )
        return None

    def _secondary_recognize_lines(
        self,
        images: Sequence[Image.Image],
    ) -> list[_SecondaryRecognition | None]:
        if not images:
            return []
        model = self._secondary_recognizer()
        if model is None:
            return [None] * len(images)

        arrays = [np.asarray(image.convert("RGB")) for image in images]

        def predict(current_model: Any) -> list[Any]:
            return list(
                current_model.predict(
                    input=arrays,
                    batch_size=self.settings.recognition_batch_size,
                )
            )

        try:
            outputs = predict(model)
        except Exception as exc:
            can_fallback = bool(
                self.settings.paddle_mkldnn_fallback
                and self._secondary_mkldnn_effective
                and self._is_onednn_pir_error(exc)
            )
            if can_fallback:
                safe_model = self._rebuild_secondary_without_mkldnn()
                if safe_model is not None:
                    try:
                        outputs = predict(safe_model)
                    except Exception as retry_exc:
                        self._secondary_model_error = (
                            f"{type(retry_exc).__name__}: {str(retry_exc)[:240]}"
                        )
                        return [None] * len(images)
                else:
                    return [None] * len(images)
            else:
                self._secondary_model_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                return [None] * len(images)

        parsed: list[_SecondaryRecognition | None] = []
        parse_error_count = 0
        for item in outputs:
            try:
                value = self._parse_secondary_item(item)
            except (TypeError, ValueError, KeyError, IndexError):
                value = None
            if value is None:
                parse_error_count += 1
            parsed.append(value)
        if parse_error_count:
            self._secondary_model_error = (
                f"secondary_parse_error: {parse_error_count} malformed result(s)"
            )
        if len(parsed) < len(images):
            parsed.extend([None] * (len(images) - len(parsed)))
        return parsed[: len(images)]

    def _secondary_recognize_line(self, image: Image.Image) -> _SecondaryRecognition | None:
        values = self._secondary_recognize_lines([image])
        return values[0] if values else None

    def _apply_semantic_verification(
        self,
        crops: Sequence[LineCrop],
        recognitions: Sequence[Recognition],
        page_metrics: dict[int, dict[str, Any]],
        candidate_indices: Sequence[int] | None = None,
        retry_crops: Sequence[LineCrop] | None = None,
    ) -> list[Recognition]:
        recognitions = tuple(
            _normalize_recognition_legal_collocations(recognition)
            for recognition in recognitions
        )
        valid_candidates = (
            set(range(min(len(crops), len(recognitions))))
            if candidate_indices is None
            else {
                int(index)
                for index in candidate_indices
                if 0 <= int(index) < min(len(crops), len(recognitions))
            }
        )
        page_totals: dict[int, int] = {}
        page_candidates: dict[int, int] = {}
        for index, crop in enumerate(crops):
            page = _page_index(crop.crop_id)
            page_totals[page] = page_totals.get(page, 0) + 1
            if index in valid_candidates:
                page_candidates[page] = page_candidates.get(page, 0) + 1

        for page, total in page_totals.items():
            metrics = page_metrics.setdefault(page, {})
            metrics.setdefault("semantic_verified_count", 0)
            metrics.setdefault("semantic_secondary_unavailable_count", 0)
            metrics.setdefault("semantic_secondary_error_count", 0)
            metrics.setdefault("semantic_auto_trimmed_count", 0)
            metrics.setdefault("semantic_high_risk_count", 0)
            metrics.setdefault("semantic_consensus_retry_attempted_count", 0)
            metrics.setdefault("semantic_consensus_retry_count", 0)
            metrics.setdefault("semantic_consensus_retry_eligible_count", 0)
            metrics.setdefault("semantic_consensus_retry_events", [])
            metrics.setdefault("semantic_surface_consensus_count", 0)
            metrics.setdefault("semantic_verifier_consensus_count", 0)
            metrics.setdefault("semantic_verification_ms", 0.0)
            metrics["semantic_candidate_count"] = (
                metrics.get("semantic_candidate_count", 0)
                + page_candidates.get(page, 0)
            )
            metrics["semantic_skipped_count"] = (
                metrics.get("semantic_skipped_count", 0)
                + total
                - page_candidates.get(page, 0)
            )

        started = time.perf_counter()
        ordered_candidates = sorted(valid_candidates)
        selected_values = (
            self._secondary_recognize_lines(
                [crops[index].image for index in ordered_candidates]
            )
            if self.settings.semantic_verification_enabled
            and self.settings.secondary_recognizer_enabled
            and ordered_candidates
            else [None] * len(ordered_candidates)
        )
        secondary_values = dict(zip(ordered_candidates, selected_values))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for page, count in page_candidates.items():
            metrics = page_metrics.setdefault(page, {})
            metrics["semantic_verification_ms"] = (
                metrics.get("semantic_verification_ms", 0.0)
                + elapsed_ms * count / max(len(ordered_candidates), 1)
            )
            if self._secondary_model_error:
                metrics["semantic_secondary_error_count"] += 1

        normalized_recognitions = list(recognitions)
        for index in ordered_candidates:
            recognition = normalized_recognitions[index]
            secondary = secondary_values.get(index)
            if (
                secondary is None
                or not recognition.verifier_text
                or recognition.verifier_confidence is None
                or recognition.verifier_confidence
                < self.settings.semantic_retry_min_verifier_confidence
                or secondary.confidence
                < self.settings.semantic_retry_min_secondary_confidence
            ):
                continue
            normalized_text = restore_three_engine_separators(
                recognition.text,
                recognition.verifier_text,
                secondary.text,
            )
            if normalized_text == recognition.text:
                continue
            page = _page_index(crops[index].crop_id)
            page_metrics.setdefault(page, {})["semantic_surface_consensus_count"] += 1
            normalized_recognitions[index] = (
                _revalidate_tesseract_after_separator_consensus(
                    recognition,
                    normalized_text,
                    crops[index],
                    self.settings,
                )
            )
            normalized_recognitions[index].secondary_confidence = secondary.confidence
        recognitions = tuple(normalized_recognitions)

        verifier_consensus_recognitions = list(recognitions)
        for index in ordered_candidates:
            recognition = verifier_consensus_recognitions[index]
            secondary = secondary_values.get(index)
            if recognition.error_code is not None:
                continue
            consensus = choose_verifier_consensus(
                primary_text=recognition.text,
                verifier_text=recognition.verifier_text,
                verifier_confidence=recognition.verifier_confidence,
                secondary_text=secondary.text if secondary is not None else None,
                secondary_confidence=(
                    secondary.confidence if secondary is not None else None
                ),
                settings=self.settings,
            )
            if not consensus.applied or secondary is None:
                continue
            page = _page_index(crops[index].crop_id)
            page_metrics.setdefault(page, {})["semantic_verifier_consensus_count"] += 1
            verifier_consensus_recognitions[index] = Recognition(
                text=consensus.text,
                confidence=(
                    consensus.confidence
                    if consensus.confidence is not None
                    else recognition.confidence
                ),
                error_code=None,
                message_vi=(
                    "Đã thay dòng bằng kết quả Tesseract được PaddleOCR độc lập "
                    "xác nhận về cấu trúc"
                ),
                raw_text=recognition.raw_text or recognition.text,
                semantic_risk="medium",
                semantic_reasons=(consensus.reason,),
                secondary_confidence=secondary.confidence,
                verifier_text=recognition.verifier_text,
                verifier_confidence=recognition.verifier_confidence,
            )
        recognitions = tuple(verifier_consensus_recognitions)

        provisional_decisions = {}
        retry_candidate_indices: list[int] = []
        additional_reasons_by_index: dict[int, Sequence[str]] = {}
        for index in ordered_candidates:
            recognition = recognitions[index]
            secondary = secondary_values.get(index)
            if secondary is not None:
                provisional = evaluate_semantic_line(
                    primary_text=recognition.text,
                    primary_confidence=recognition.confidence,
                    primary_error_code=recognition.error_code,
                    secondary_text=secondary.text,
                    secondary_confidence=secondary.confidence,
                    settings=self.settings,
                )
                provisional_decisions[index] = provisional
                additional_reasons_by_index[index] = provisional.reasons
            if (
                self.settings.semantic_retry_enabled
                and secondary is not None
                and recognition.verifier_text
                and recognition.verifier_confidence is not None
                and recognition.verifier_confidence
                >= self.settings.semantic_retry_min_verifier_confidence
                and secondary.confidence
                >= self.settings.semantic_retry_min_secondary_confidence
                and (
                    recognition.error_code is not None
                    or recognition.confidence
                    <= self.settings.semantic_retry_primary_max_confidence
                )
            ):
                retry_candidate_indices.append(index)

        for index in retry_candidate_indices:
            page = _page_index(crops[index].crop_id)
            metrics = page_metrics.setdefault(page, {})
            metrics["semantic_consensus_retry_eligible_count"] = (
                metrics.get("semantic_consensus_retry_eligible_count", 0) + 1
            )

        semantic_retry_indices = _prioritize_semantic_retry_indices(
            crops,
            recognitions,
            candidate_indices=retry_candidate_indices,
            max_per_page=self.settings.semantic_retry_max_lines_per_page,
            additional_reasons_by_index=additional_reasons_by_index,
        )
        retry_variant_inputs: list[LineCrop] = []
        retry_variant_indices: list[int] = []
        for index in semantic_retry_indices:
            retry_crop = (
                retry_crops[index]
                if retry_crops is not None and index < len(retry_crops)
                else crops[index]
            )
            retry_variant_indices.append(index)
            retry_variant_inputs.append(retry_crop)
        retry_batch_started = time.perf_counter()
        retry_variant_values = self._semantic_retry_variants_many(retry_variant_inputs)
        retry_batch_elapsed_ms = (time.perf_counter() - retry_batch_started) * 1000.0
        retry_variants_by_index = dict(zip(retry_variant_indices, retry_variant_values))
        retry_elapsed_by_index = {
            index: round(retry_batch_elapsed_ms / max(len(retry_variant_indices), 1), 3)
            for index in retry_variant_indices
        }
        verified: list[Recognition] = []
        for index, recognition in enumerate(recognitions):
            if index not in valid_candidates:
                verified.append(recognition)
                continue
            crop = crops[index] if index < len(crops) else None
            page = _page_index(crop.crop_id) if crop is not None else 0
            metrics = page_metrics.setdefault(page, {})
            secondary = secondary_values.get(index)
            if secondary is None:
                metrics["semantic_secondary_unavailable_count"] += 1
            else:
                metrics["semantic_verified_count"] += 1
            retry_decision = None
            retry_eligible = index in semantic_retry_indices
            retry_crop = (
                retry_crops[index]
                if retry_crops is not None and index < len(retry_crops)
                else crop
            )
            if retry_eligible and retry_crop is not None:
                metrics["semantic_consensus_retry_attempted_count"] += 1
                retry_variants = retry_variants_by_index.get(index, ())
                retry_decision = choose_consensus_retry(
                    primary_text=recognition.text,
                    verifier_text=recognition.verifier_text,
                    verifier_confidence=recognition.verifier_confidence,
                    secondary_text=secondary.text,
                    secondary_confidence=secondary.confidence,
                    variants=retry_variants,
                    settings=self.settings,
                )
                retry_elapsed_ms = retry_elapsed_by_index.get(index, 0.0)
                metrics["semantic_consensus_retry_events"].append(
                    {
                        "crop_id": crop.crop_id,
                        "applied": retry_decision.applied,
                        "reason": retry_decision.reason,
                        "selected_width": retry_decision.selected_width,
                        "confidence": retry_decision.confidence,
                        "elapsed_ms": retry_elapsed_ms,
                        "before_text": recognition.text,
                        "after_text": (
                            retry_decision.text
                            if retry_decision.applied
                            else recognition.text
                        ),
                        "primary_confidence": recognition.confidence,
                        "verifier_text": recognition.verifier_text,
                        "verifier_confidence": recognition.verifier_confidence,
                        "secondary_text": secondary.text,
                        "secondary_confidence": secondary.confidence,
                        "variants": [
                            {
                                "text": variant.text,
                                "confidence": variant.confidence,
                                "resized_width": variant.resized_width,
                            }
                            for variant in retry_variants
                        ],
                    }
                )
            if retry_decision is not None and retry_decision.applied:
                metrics["semantic_consensus_retry_count"] += 1
                verified.append(
                    Recognition(
                        text=retry_decision.text,
                        confidence=(
                            retry_decision.confidence
                            if retry_decision.confidence is not None
                            else recognition.confidence
                        ),
                        error_code=None,
                        message_vi=(
                            "Đã nhận dạng lại dòng ngắn hơn và xác nhận bằng "
                            "hai OCR độc lập"
                        ),
                        raw_text=recognition.raw_text or recognition.text,
                        semantic_risk="medium",
                        semantic_reasons=(retry_decision.reason,),
                        secondary_confidence=secondary.confidence,
                        verifier_text=recognition.verifier_text,
                        verifier_confidence=recognition.verifier_confidence,
                    )
                )
                continue
            decision = provisional_decisions.get(index)
            if decision is None:
                decision = evaluate_semantic_line(
                    primary_text=recognition.text,
                    primary_confidence=recognition.confidence,
                    primary_error_code=recognition.error_code,
                    secondary_text=secondary.text if secondary is not None else None,
                    secondary_confidence=(
                        secondary.confidence if secondary is not None else None
                    ),
                    settings=self.settings,
                )
            risk_order = {"none": 0, "medium": 1, "high": 2}
            combined_risk = max(
                (recognition.semantic_risk, decision.risk),
                key=risk_order.__getitem__,
            )
            combined_reasons = tuple(
                dict.fromkeys((*recognition.semantic_reasons, *decision.reasons))
            )
            if combined_risk == "high":
                metrics["semantic_high_risk_count"] += 1
            if "unsupported_suffix_removed" in decision.reasons:
                metrics["semantic_auto_trimmed_count"] += 1
            verified.append(
                Recognition(
                    text=decision.text,
                    confidence=recognition.confidence,
                    error_code=recognition.error_code,
                    message_vi=recognition.message_vi,
                    raw_text=decision.raw_text or recognition.raw_text,
                    semantic_risk=combined_risk,
                    semantic_reasons=combined_reasons,
                    secondary_confidence=(
                        decision.secondary_confidence
                        if decision.secondary_confidence is not None
                        else recognition.secondary_confidence
                    ),
                    verifier_text=recognition.verifier_text,
                    verifier_confidence=recognition.verifier_confidence,
                )
            )
        return verified

    def remediate_high_risk_candidates(
        self,
        crops: Sequence[LineCrop],
        recognitions: Sequence[Recognition],
        page_metrics: dict[int, dict[str, Any]],
    ) -> list[Recognition]:
        """Một lượt cứu hộ cuối; chỉ thay text khi ba engine đủ đồng thuận."""
        started = time.perf_counter()
        secondaries = self._secondary_recognize_lines(
            [crop.image for crop in crops[: len(recognitions)]]
        )
        retry_indices = [
            index
            for index, (recognition, secondary) in enumerate(
                zip(recognitions, secondaries)
            )
            if secondary is not None and recognition.verifier_text
        ]
        retry_started = time.perf_counter()
        retry_values = self._semantic_retry_variants_many(
            [crops[index] for index in retry_indices]
        )
        retry_elapsed_ms = (time.perf_counter() - retry_started) * 1000.0
        retry_variants_by_index = dict(zip(retry_indices, retry_values))
        repaired: list[Recognition] = []
        for index, (crop, recognition, secondary) in enumerate(
            zip(crops, recognitions, secondaries)
        ):
            page = _page_index(crop.crop_id)
            metrics = page_metrics.setdefault(page, {})
            metrics["attempted_count"] = metrics.get("attempted_count", 0) + 1
            metrics.setdefault("applied_count", 0)
            metrics.setdefault("events", [])
            variants = retry_variants_by_index.get(index, ())
            decision = choose_consensus_retry(
                primary_text=recognition.text,
                verifier_text=recognition.verifier_text,
                verifier_confidence=recognition.verifier_confidence,
                secondary_text=secondary.text if secondary is not None else None,
                secondary_confidence=(
                    secondary.confidence if secondary is not None else None
                ),
                variants=variants,
                settings=self.settings,
            )
            elapsed_ms = round(
                retry_elapsed_ms / max(len(retry_indices), 1),
                3,
            ) if index in retry_variants_by_index else 0.0
            metrics["events"].append(
                {
                    "crop_id": crop.crop_id,
                    "applied": decision.applied,
                    "reason": decision.reason,
                    "selected_width": decision.selected_width,
                    "confidence": decision.confidence,
                    "elapsed_ms": elapsed_ms,
                    "before_text": recognition.text,
                    "after_text": (
                        decision.text if decision.applied else recognition.text
                    ),
                    "primary_confidence": recognition.confidence,
                    "verifier_text": recognition.verifier_text,
                    "verifier_confidence": recognition.verifier_confidence,
                    "secondary_text": secondary.text if secondary is not None else None,
                    "secondary_confidence": (
                        secondary.confidence if secondary is not None else None
                    ),
                    "variants": [
                        {
                            "text": variant.text,
                            "confidence": variant.confidence,
                            "resized_width": variant.resized_width,
                        }
                        for variant in variants
                    ],
                }
            )
            if not decision.applied:
                repaired.append(recognition)
                continue
            metrics["applied_count"] += 1
            repaired.append(
                Recognition(
                    text=decision.text,
                    confidence=(
                        decision.confidence
                        if decision.confidence is not None
                        else recognition.confidence
                    ),
                    error_code=None,
                    message_vi=(
                        "Đã cứu hộ dòng rủi ro bằng đồng thuận VietOCR, "
                        "PaddleOCR và Tesseract dòng đơn"
                    ),
                    raw_text=recognition.raw_text or recognition.text,
                    semantic_risk="medium",
                    semantic_reasons=(
                        "partial_remediation_consensus_applied",
                        decision.reason,
                    ),
                    secondary_confidence=(
                        secondary.confidence if secondary is not None else None
                    ),
                    verifier_text=recognition.verifier_text,
                    verifier_confidence=recognition.verifier_confidence,
                )
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        counts_by_page: dict[int, int] = {}
        for crop in crops[: len(recognitions)]:
            page = _page_index(crop.crop_id)
            counts_by_page[page] = counts_by_page.get(page, 0) + 1
        for page, count in counts_by_page.items():
            page_metrics[page]["elapsed_ms"] = round(
                elapsed_ms * count / max(len(crops), 1),
                3,
            )
        return repaired

    def _semantic_retry_variants(self, crop: LineCrop) -> tuple[RetryVariant, ...]:
        predictor = self._predictor()
        widths: list[int] = []
        for value in self.settings.semantic_retry_widths.split(","):
            try:
                width = int(value.strip())
            except ValueError:
                continue
            if 96 <= width <= 1024 and width not in widths:
                widths.append(width)

        variants: list[RetryVariant] = []
        for width in widths:
            split_images = _split_wide_image(
                crop.image,
                image_height=self.settings.vietocr_image_height,
                image_max_width=width,
                target_ratio=self.settings.wide_crop_target_ratio,
                max_segments=self.settings.wide_crop_max_segments,
                valley_max_ink_ratio=self.settings.wide_crop_valley_max_ink_ratio,
                overlap_height_ratio=0.0,
            )
            if len(split_images) <= 1:
                continue
            parts = [
                self._recognize_one(predictor, split_image.image)
                for split_image in split_images
            ]
            if not parts or any(not part.text for part in parts):
                continue
            text, _ = _join_segments(
                [part.text for part in parts],
                max_token_overlap=self.settings.seam_max_token_overlap,
                max_char_overlap=self.settings.seam_max_char_overlap,
            )
            variants.append(
                RetryVariant(
                    text=text,
                    confidence=min(part.confidence for part in parts),
                    resized_width=width,
                )
            )
        return tuple(variants)

    def _semantic_retry_cache_key(self, crop: LineCrop) -> str:
        image = crop.image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        digest.update(
            (
                f"{self.settings.semantic_retry_widths}|"
                f"{self.settings.vietocr_image_height}|"
                f"{self.settings.wide_crop_target_ratio}|"
                f"{self.settings.wide_crop_max_segments}|"
                f"{self.settings.wide_crop_valley_max_ink_ratio}|"
                f"{self.settings.seam_max_token_overlap}|"
                f"{self.settings.seam_max_char_overlap}"
            ).encode("ascii")
        )
        return digest.hexdigest()

    def _semantic_retry_variants_many(
        self,
        crops: Sequence[LineCrop],
    ) -> list[tuple[RetryVariant, ...]]:
        """Reuse exact sequential retry evidence within one OCR window."""
        results: list[tuple[RetryVariant, ...]] = []
        for crop in crops:
            key = self._semantic_retry_cache_key(crop)
            cached = self._semantic_retry_cache.get(key)
            if cached is None:
                cached = self._semantic_retry_variants(crop)
                self._semantic_retry_cache[key] = cached
            results.append(cached)
        return results

    def verify_semantic_candidates(
        self,
        crops: Sequence[LineCrop],
        recognitions: Sequence[Recognition],
        candidate_indices: Sequence[int],
        page_metrics: dict[int, dict[str, Any]],
        retry_crops: Sequence[LineCrop] | None = None,
    ) -> list[Recognition]:
        """Run the secondary OCR only for risk-selected line crops."""
        return self._apply_semantic_verification(
            crops,
            recognitions,
            page_metrics,
            candidate_indices,
            retry_crops,
        )

    def _prepare_segments(
        self,
        crops: Sequence[LineCrop],
        predictor: Any,
    ) -> tuple[list[_Segment], list[list[int]], dict[str, Any], set[int]]:
        image_height, image_max_width = _predictor_dimensions(predictor, self.settings)
        segments: list[_Segment] = []
        by_original: list[list[int]] = [[] for _ in crops]
        wide_crop_count = 0
        widths_before: list[int] = []
        widths_after: list[int] = []
        width_cap_detected: set[int] = set()
        width_cap_resolved: set[int] = set()
        width_cap_unresolved: set[int] = set()
        overlap_pixel_total = 0

        for original_index, crop in enumerate(crops):
            before = _resized_width(crop.image, image_height)
            widths_before.append(before)
            full_ink, full_ink_columns = _image_ink_stats(crop.image)
            split_images = [
                _SplitImage(
                    image=crop.image.convert("RGB"),
                    core_left=0,
                    core_right=crop.image.width,
                    source_left=0,
                    source_right=crop.image.width,
                    left_overlap=0,
                    right_overlap=0,
                    ink_ratio=full_ink,
                    ink_columns_ratio=full_ink_columns,
                    is_tail=True,
                )
            ]
            is_width_capped = before > image_max_width
            if is_width_capped:
                width_cap_detected.add(original_index)
            should_attempt_split = (
                self.settings.split_wide_crops
                and (
                    (self.settings.split_width_capped_crops and is_width_capped)
                    or before
                    > image_max_width * self.settings.wide_crop_split_threshold_ratio
                )
            )
            if should_attempt_split:
                split_images = _split_wide_image(
                    crop.image,
                    image_height,
                    image_max_width,
                    self.settings.wide_crop_target_ratio,
                    self.settings.wide_crop_max_segments,
                    self.settings.wide_crop_valley_max_ink_ratio,
                    (
                        self.settings.wide_crop_overlap_height_ratio
                        if self.settings.experimental_overlap_split
                        else 0.0
                    ),
                )
                if len(split_images) > 1:
                    wide_crop_count += 1
            if is_width_capped:
                if len(split_images) > 1 and all(
                    _resized_width(item.image, image_height) <= image_max_width
                    for item in split_images
                ):
                    width_cap_resolved.add(original_index)
                else:
                    width_cap_unresolved.add(original_index)
            page_index = _page_index(crop.crop_id)
            for segment_index, split_image in enumerate(split_images):
                resized = _resized_width(split_image.image, image_height)
                widths_after.append(resized)
                overlap_pixel_total += split_image.left_overlap + split_image.right_overlap
                index = len(segments)
                segments.append(
                    _Segment(
                        original_index=original_index,
                        segment_index=segment_index,
                        page_index=page_index,
                        image=split_image.image,
                        resized_width=resized,
                        was_split=len(split_images) > 1,
                        source_left=split_image.source_left,
                        source_right=split_image.source_right,
                        left_overlap=split_image.left_overlap,
                        right_overlap=split_image.right_overlap,
                        ink_ratio=split_image.ink_ratio,
                        ink_columns_ratio=split_image.ink_columns_ratio,
                        is_tail=split_image.is_tail,
                    )
                )
                by_original[original_index].append(index)

        return segments, by_original, {
            "wide_crop_count": wide_crop_count,
            "segment_count": len(segments),
            "split_segment_count": max(0, len(segments) - len(crops)),
            "split_overlap_source_pixels": overlap_pixel_total,
            "max_resized_width_before_split": max(widths_before, default=0),
            "max_resized_width_after_split": max(widths_after, default=0),
            "width_cap_detected_count": len(width_cap_detected),
            "width_cap_resolved_count": len(width_cap_resolved),
            "width_cap_unresolved_count": len(width_cap_unresolved),
            "width_capped_after_split_count": sum(
                width > image_max_width for width in widths_after
            ),
        }, width_cap_unresolved

    def recognize(self, crops: Sequence[LineCrop]) -> list[Recognition]:
        # Retry evidence is reusable only inside the current OCR window. The
        # persistent engine must not retain page images across requests.
        self._semantic_retry_cache.clear()
        diagnostics_overhead_ms = 0.0
        if self.settings.performance_diagnostics_enabled:
            diagnostics_started = time.perf_counter()
            runtime_start = _runtime_snapshot()
            diagnostics_overhead_ms += (time.perf_counter() - diagnostics_started) * 1000.0
        else:
            runtime_start = {}
        recognize_wall_started = time.perf_counter()
        recognize_cpu_started = time.process_time()
        if not crops:
            self.last_batch_metrics = {
                "crop_count": 0,
                "segment_count": 0,
                "batch_count": 0,
                "mean_batch_size": 0.0,
                "max_batch_size": 0,
                "beam_retry_count": 0,
            }
            self.last_page_metrics = {}
            return []

        predictor_started = time.perf_counter()
        predictor_cpu_started = time.process_time()
        predictor = self._predictor()
        predictor_resolve_wall_ms = (time.perf_counter() - predictor_started) * 1000.0
        predictor_resolve_cpu_ms = (time.process_time() - predictor_cpu_started) * 1000.0
        segments, by_original, split_metrics, width_cap_unresolved = self._prepare_segments(
            crops, predictor
        )
        segment_texts = [""] * len(segments)
        segment_probabilities = [0.0] * len(segments)
        # Giữ đúng ảnh đã đưa vào batch (bao gồm common-width padding) để trace
        # decoder tái lập cùng output, tránh mismatch do chạy lại trên ảnh khác.
        inference_images = [segment.image for segment in segments]
        batch_count = 0
        max_batch_size = 0
        batch_sizes: list[int] = []
        greedy_ms = 0.0
        greedy_cpu_ms = 0.0
        padded_segment_count = 0
        padding_source_columns = 0
        normalized_batch_widths: list[int] = []
        page_metrics: dict[int, dict[str, Any]] = {}

        image_height, image_max_width = _predictor_dimensions(predictor, self.settings)
        for page_index in {_page_index(crop.crop_id) for crop in crops}:
            page_crop_indices = [
                index
                for index, crop in enumerate(crops)
                if _page_index(crop.crop_id) == page_index
            ]
            page_segment_indices = [
                index
                for index, segment in enumerate(segments)
                if segment.page_index == page_index
            ]
            page_metrics[page_index] = {
                "crop_count": len(page_crop_indices),
                "segment_count": len(page_segment_indices),
                "split_segment_count": max(0, len(page_segment_indices) - len(page_crop_indices)),
                "split_overlap_source_pixels": sum(
                    segments[index].left_overlap + segments[index].right_overlap
                    for index in page_segment_indices
                ),
                "wide_crop_count": sum(
                    1 for index in page_crop_indices if len(by_original[index]) > 1
                ),
                "width_cap_detected_count": sum(
                    _resized_width(crops[index].image, image_height) > image_max_width
                    for index in page_crop_indices
                ),
                "width_cap_resolved_count": sum(
                    _resized_width(crops[index].image, image_height) > image_max_width
                    and index not in width_cap_unresolved
                    for index in page_crop_indices
                ),
                "width_cap_unresolved_count": sum(
                    index in width_cap_unresolved for index in page_crop_indices
                ),
                "max_resized_width_before_split": max(
                    (
                        _resized_width(crops[index].image, image_height)
                        for index in page_crop_indices
                    ),
                    default=0,
                ),
                "max_resized_width_after_split": max(
                    (segments[index].resized_width for index in page_segment_indices),
                    default=0,
                ),
                "width_capped_after_split_count": sum(
                    segments[index].resized_width > image_max_width
                    for index in page_segment_indices
                ),
                "shared_batch_count": 0,
                "padded_segment_count": 0,
                "padding_source_columns": 0,
                "greedy_batch_ms_allocated": 0.0,
                "greedy_model_cpu_ms_allocated": 0.0,
                "greedy_batch_fallback_count": 0,
                "greedy_batch_fallback_segment_count": 0,
                "greedy_batch_fallback_error_types": {},
                "decoder_preprocess_wall_ms": 0.0,
                "decoder_preprocess_cpu_ms": 0.0,
                "decoder_encoder_wall_ms": 0.0,
                "decoder_encoder_cpu_ms": 0.0,
                "decoder_model_wall_ms": 0.0,
                "decoder_model_cpu_ms": 0.0,
                "decoder_attention_extract_wall_ms": 0.0,
                "decoder_attention_extract_cpu_ms": 0.0,
                "decoder_torch_postprocess_wall_ms": 0.0,
                "decoder_torch_postprocess_cpu_ms": 0.0,
                "decoder_visual_grounding_wall_ms": 0.0,
                "decoder_visual_grounding_cpu_ms": 0.0,
                "decoder_trace_build_wall_ms": 0.0,
                "decoder_trace_build_cpu_ms": 0.0,
                "decoder_forward_call_count": 0,
                "decoder_sample_step_count": 0,
                "decoder_attention_element_count": 0,
                "decoder_trace_input_pixel_count": 0,
                "decoder_trace_character_count": 0,
                "cross_segment_analysis_ms": 0.0,
                "beam_retry_count": 0,
                "beam_retry_ms": 0.0,
                "tail_segment_count": sum(
                    segments[index].was_split and segments[index].is_tail
                    for index in page_segment_indices
                ),
                "tail_segment_retry_count": 0,
                "tail_segment_retry_accepted_count": 0,
                "tail_segment_uncertain_count": 0,
                "tail_segment_suppressed_count": 0,
                "tail_segment_retry_ms": 0.0,
                "seam_overlap_merge_count": 0,
                "decoder_loop_detected_count": 0,
                "decoder_partial_loop_detected_count": 0,
                "decoder_loop_trimmed_count": 0,
                "decoder_char_loop_detected_count": 0,
                "decoder_char_loop_trimmed_count": 0,
                "empty_recognition_count": 0,
                "hallucination_guard_candidate_count": 0,
                "hallucination_guard_retry_count": 0,
                "hallucination_guard_disagreement_count": 0,
                "hallucination_guard_consensus_count": 0,
                "hallucination_guard_removed_token_count": 0,
                "hallucination_guard_suffix_removed_count": 0,
                "hallucination_guard_prefix_removed_count": 0,
                "hallucination_guard_midline_removed_count": 0,
                "hallucination_guard_numeric_removed_count": 0,
                "hallucination_guard_dominant_ink_retry_count": 0,
                "hallucination_guard_ms": 0.0,
                "hallucination_guard_events": [],
                "decoder_evidence_candidate_count": 0,
                "decoder_evidence_seed_selected_count": 0,
                "decoder_evidence_context_forced_count": 0,
                "decoder_evidence_selected_count": 0,
                "decoder_evidence_unchecked_candidate_count": 0,
                "decoder_evidence_trace_count": 0,
                "decoder_evidence_trace_batch_count": 0,
                "decoder_evidence_trace_batch_size_max": 0,
                "decoder_evidence_supported_count": 0,
                "decoder_evidence_trace_mismatch_count": 0,
                "decoder_evidence_trace_error_count": 0,
                "decoder_evidence_trace_error_types": {},
                "decoder_evidence_disabled_reason": None,
                "decoder_evidence_circuit_breaker_count": 0,
                "decoder_evidence_attention_stall_count": 0,
                "decoder_evidence_visual_exhausted_count": 0,
                "decoder_evidence_near_loop_count": 0,
                "decoder_evidence_midline_span_count": 0,
                "decoder_evidence_midline_trimmed_count": 0,
                "decoder_evidence_line_evidence_count": 0,
                "decoder_evidence_cross_segment_candidate_count": 0,
                "decoder_evidence_cross_segment_trimmed_count": 0,
                "decoder_evidence_cross_segment_rejected_count": 0,
                "secondary_verifier_count": 0,
                "secondary_verifier_primary_extra_count": 0,
                "secondary_verifier_conflict_count": 0,
                "secondary_verifier_ambiguous_count": 0,
                "secondary_verifier_error_count": 0,
                "secondary_verifier_ms": 0.0,
                "decoder_evidence_visual_coverage_exhausted_count": 0,
                "decoder_evidence_attention_reuse_count": 0,
                "decoder_evidence_trimmed_count": 0,
                "decoder_evidence_trimmed_char_count": 0,
                "decoder_evidence_suspicious_numeric_count": 0,
                "decoder_evidence_cluster_expansion_count": 0,
                "decoder_evidence_expanded_word_count": 0,
                "decoder_evidence_ms": 0.0,
                "decoder_evidence_events": [],
                "decoder_evidence_cross_segment_rejections": [],
            }

        for batch_indices in _adaptive_batches(
            segments,
            self.settings.recognition_batch_size,
            self.settings.recognition_pixel_budget,
            self.settings.recognition_batch_width_ratio,
        ):
            batch_count += 1
            max_batch_size = max(max_batch_size, len(batch_indices))
            batch_sizes.append(len(batch_indices))
            target_width = min(
                image_max_width,
                max(segments[index].resized_width for index in batch_indices),
            )
            images: list[Image.Image] = []
            padding_by_index: dict[int, int] = {}
            for index in batch_indices:
                image = segments[index].image
                padding = 0
                if self.settings.pad_batches_to_common_width:
                    image, padding = _pad_to_resized_width(
                        image,
                        target_width,
                        image_height,
                    )
                images.append(image)
                inference_images[index] = image
                padding_by_index[index] = padding
                if padding:
                    page = segments[index].page_index
                    page_metrics[page]["padded_segment_count"] += 1
                    page_metrics[page]["padding_source_columns"] += padding
            padded_segment_count += sum(1 for value in padding_by_index.values() if value)
            padding_source_columns += sum(padding_by_index.values())
            normalized_batch_widths.append(target_width)
            started = time.perf_counter()
            cpu_started = time.process_time()
            try:
                if not hasattr(predictor, "predict_batch"):
                    raise AttributeError("predict_batch unavailable")
                with _inference_context(self.settings.use_torch_inference_mode):
                    texts, probabilities = _unpack_batch(
                        predictor.predict_batch(images, return_prob=True)
                    )
                if len(texts) != len(images) or len(probabilities) != len(images):
                    raise ValueError("Số kết quả VietOCR không khớp số ảnh dòng")
                for index, text, probability in zip(
                    batch_indices, texts, probabilities, strict=True
                ):
                    segment_texts[index] = normalize_vietnamese_text(str(text))
                    segment_probabilities[index] = float(probability)
            except Exception as exc:
                error_type = type(exc).__name__
                fallback_pages: dict[int, int] = {}
                for index in batch_indices:
                    page = segments[index].page_index
                    fallback_pages[page] = fallback_pages.get(page, 0) + 1
                    recognition = self._recognize_one(predictor, segments[index].image)
                    segment_texts[index] = recognition.text
                    segment_probabilities[index] = recognition.confidence
                for page, count in fallback_pages.items():
                    page_metrics[page]["greedy_batch_fallback_count"] += 1
                    page_metrics[page]["greedy_batch_fallback_segment_count"] += count
                    error_types = page_metrics[page]["greedy_batch_fallback_error_types"]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            elapsed_ms = (time.perf_counter() - started) * 1000
            elapsed_cpu_ms = (time.process_time() - cpu_started) * 1000
            greedy_ms += elapsed_ms
            greedy_cpu_ms += elapsed_cpu_ms
            pages_in_batch: dict[int, int] = {}
            for index in batch_indices:
                page = segments[index].page_index
                pages_in_batch[page] = pages_in_batch.get(page, 0) + 1
            for page, count in pages_in_batch.items():
                page_metrics[page]["shared_batch_count"] += 1
                page_metrics[page]["greedy_batch_ms_allocated"] += (
                    elapsed_ms * count / len(batch_indices)
                )
                page_metrics[page]["greedy_model_cpu_ms_allocated"] += (
                    elapsed_cpu_ms * count / len(batch_indices)
                )

        # Decoder-evidence guard. Các candidate cùng kích thước được trace theo
        # batch để mở rộng coverage mà không lặp encoder/decoder cho từng dòng.
        decoder_changed_indices: set[int] = set()
        numeric_warning_segment_indices: set[int] = set()
        decoder_evidence_ms_total = 0.0
        decoder_evidence_disabled_reason: str | None = None
        decoder_traces: dict[int, _DecoderEvidenceTrace] = {}
        decoder_scores: dict[int, float] = {}
        decoder_seed_indices: set[int] = set()
        if self.settings.decoder_evidence_enabled:
            decoder_candidates: dict[int, list[tuple[float, int]]] = {}
            decoder_all_scores: dict[int, float] = {}
            for index, segment in enumerate(segments):
                score = _decoder_evidence_candidate_score(
                    segment,
                    segment_texts[index],
                    segment_probabilities[index],
                    self.settings,
                )
                decoder_all_scores[index] = score
                if (
                    segment_texts[index]
                    and (
                        self.settings.decoder_evidence_full_coverage
                        or score >= self.settings.decoder_evidence_candidate_score_threshold
                    )
                ):
                    decoder_candidates.setdefault(segment.page_index, []).append((score, index))

            for page in sorted(decoder_candidates):
                values = decoder_candidates[page]
                values.sort(key=lambda item: item[0], reverse=True)
                check_limit = self.settings.decoder_evidence_max_checks_per_page
                seed_selected = (
                    list(values)
                    if self.settings.decoder_evidence_full_coverage or check_limit == 0
                    else list(values[:check_limit])
                )
                decoder_seed_indices.update(index for _, index in seed_selected)
                selected_by_index = {index: score for score, index in seed_selected}
                if (
                    not self.settings.decoder_evidence_full_coverage
                    and self.settings.decoder_evidence_include_split_line_context
                    and self.settings.decoder_evidence_cross_segment_enabled
                ):
                    # A top-K candidate on a split line is not useful for
                    # cross-segment validation without its visual anchors. Trace
                    # all segments of that line as context, but do not consume
                    # additional top-K seed slots. Typical legal lines have two.
                    for _, index in seed_selected:
                        segment = segments[index]
                        if not segment.was_split:
                            continue
                        for context_index in by_original[segment.original_index]:
                            if (
                                segments[context_index].page_index == page
                                and segment_texts[context_index]
                            ):
                                selected_by_index.setdefault(
                                    context_index, decoder_all_scores[context_index]
                                )
                selected = sorted(
                    ((score, index) for index, score in selected_by_index.items()),
                    key=lambda item: (-item[0], item[1]),
                )
                page_metrics[page]["decoder_evidence_candidate_count"] = len(values)
                page_metrics[page]["decoder_evidence_seed_selected_count"] = len(seed_selected)
                page_metrics[page]["decoder_evidence_context_forced_count"] = max(
                    0, len(selected) - len(seed_selected)
                )
                page_metrics[page]["decoder_evidence_selected_count"] = len(selected)
                page_metrics[page]["decoder_evidence_unchecked_candidate_count"] = max(
                    0, len(values) - len(seed_selected)
                )
                if decoder_evidence_disabled_reason is not None:
                    page_metrics[page]["decoder_evidence_disabled_reason"] = (
                        decoder_evidence_disabled_reason
                    )
                    continue

                # Primary greedy batches already pad to a common resized width.
                # Group by the exact resulting image size to preserve identical
                # preprocessing and avoid trace mismatches caused by repadding.
                size_groups: dict[tuple[int, int], list[tuple[float, int]]] = {}
                for score, index in selected:
                    size_groups.setdefault(inference_images[index].size, []).append(
                        (score, index)
                    )

                stop_page = False
                for group in size_groups.values():
                    for offset in range(
                        0, len(group), self.settings.decoder_evidence_trace_batch_size
                    ):
                        chunk = group[
                            offset : offset + self.settings.decoder_evidence_trace_batch_size
                        ]
                        images = [inference_images[index] for _, index in chunk]
                        started = time.perf_counter()
                        try:
                            outcomes = _trace_seq2seq_attention_batch_detailed(
                                predictor, images, self.settings, profile=page_metrics[page]
                            )
                        except TypeError as exc:
                            # Backward-compatible test/plugin adapters from <=1.3.5.4
                            # may still expose the three-argument trace hook.
                            if "profile" not in str(exc):
                                raise
                            outcomes = _trace_seq2seq_attention_batch_detailed(
                                predictor, images, self.settings
                            )
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        decoder_evidence_ms_total += elapsed_ms
                        page_metrics[page]["decoder_evidence_ms"] += elapsed_ms
                        page_metrics[page]["decoder_evidence_trace_batch_count"] += 1
                        page_metrics[page]["decoder_evidence_trace_batch_size_max"] = max(
                            page_metrics[page]["decoder_evidence_trace_batch_size_max"],
                            len(chunk),
                        )
                        page_metrics[page]["decoder_evidence_trace_count"] += len(chunk)
                        if len(outcomes) != len(chunk):
                            outcomes = [
                                _DecoderEvidenceTraceOutcome(
                                    trace=None,
                                    error_type="_DecoderContractError",
                                    error_message="batch trace outcome count mismatch",
                                    fatal=True,
                                )
                                for _ in chunk
                            ]

                        for (score, index), outcome in zip(
                            chunk, outcomes, strict=True
                        ):
                            if outcome.trace is None:
                                reason = (
                                    outcome.disabled_reason
                                    or outcome.error_type
                                    or "trace_unavailable"
                                )
                                if outcome.error_type:
                                    page_metrics[page][
                                        "decoder_evidence_trace_error_count"
                                    ] += 1
                                    error_types = page_metrics[page][
                                        "decoder_evidence_trace_error_types"
                                    ]
                                    error_types[outcome.error_type] = (
                                        error_types.get(outcome.error_type, 0) + 1
                                    )
                                page_metrics[page][
                                    "decoder_evidence_disabled_reason"
                                ] = reason
                                if outcome.fatal:
                                    decoder_evidence_disabled_reason = reason
                                    page_metrics[page][
                                        "decoder_evidence_circuit_breaker_count"
                                    ] += 1
                                    events = page_metrics[page][
                                        "decoder_evidence_events"
                                    ]
                                    if (
                                        len(events)
                                        < self.settings.decoder_evidence_event_limit
                                    ):
                                        events.append(
                                            {
                                                "segment_index": index,
                                                "risk": round(score, 3),
                                                "reason": "trace_disabled",
                                                "error_type": outcome.error_type,
                                                "error_message": outcome.error_message,
                                                "disabled_reason": reason,
                                                "applied": False,
                                            }
                                        )
                                    stop_page = True
                                    break
                                continue

                            trace = outcome.trace
                            page_metrics[page][
                                "decoder_evidence_supported_count"
                            ] += 1
                            raw_normalized = normalize_vietnamese_text(trace.raw_text)
                            if raw_normalized != segment_texts[index]:
                                page_metrics[page][
                                    "decoder_evidence_trace_mismatch_count"
                                ] += 1
                                continue
                            decoder_traces[index] = trace
                            decoder_scores[index] = score
                            # Context-forced traces exist only to provide visual
                            # anchors for a selected split line. They must not
                            # independently mutate a segment that did not pass
                            # the selective risk gate.
                            if (
                                not self.settings.decoder_evidence_full_coverage
                                and index not in decoder_seed_indices
                            ):
                                continue
                            decision = _decoder_evidence_decision(
                                trace, self.settings
                            )
                            if decision is None:
                                continue
                            if decision.reason == "attention_stall":
                                page_metrics[page][
                                    "decoder_evidence_attention_stall_count"
                                ] += 1
                            elif decision.reason == "visual_evidence_exhausted":
                                page_metrics[page][
                                    "decoder_evidence_visual_exhausted_count"
                                ] += 1
                            elif decision.reason == "attention_near_loop":
                                page_metrics[page][
                                    "decoder_evidence_near_loop_count"
                                ] += 1
                            elif decision.reason == "unsupported_midline_span":
                                page_metrics[page][
                                    "decoder_evidence_midline_span_count"
                                ] += 1
                            if decision.numeric_only_warning:
                                numeric_warning_segment_indices.add(index)
                                page_metrics[page][
                                    "decoder_evidence_suspicious_numeric_count"
                                ] += 1
                            if decision.expanded_word_count:
                                page_metrics[page][
                                    "decoder_evidence_cluster_expansion_count"
                                ] += 1
                                page_metrics[page][
                                    "decoder_evidence_expanded_word_count"
                                ] += decision.expanded_word_count

                            applied = bool(
                                self.settings.decoder_evidence_apply_changes
                                and not decision.numeric_only_warning
                            )
                            events = page_metrics[page]["decoder_evidence_events"]
                            if len(events) < self.settings.decoder_evidence_event_limit:
                                tail_count = min(
                                    self.settings.decoder_evidence_window_tokens,
                                    len(trace.token_probabilities),
                                )
                                events.append(
                                    {
                                        "segment_index": index,
                                        "risk": round(score, 3),
                                        "raw_text": segment_texts[index],
                                        "proposed_text": decision.text,
                                        "validated_text": (
                                            decision.text
                                            if applied
                                            else segment_texts[index]
                                        ),
                                        "applied": applied,
                                        "removed_text": decision.removed_text,
                                        "reason": decision.reason,
                                        "numeric_warning_only": decision.numeric_only_warning,
                                        "expanded_word_count": decision.expanded_word_count,
                                        "span_kind": decision.span_kind,
                                        "span_start": decision.span_start,
                                        "span_end": decision.span_end,
                                        "span_probability_mean": round(decision.span_probability_mean, 4),
                                        "span_ink_support_mean": round(decision.span_ink_support_mean, 4),
                                        "span_attention_range": round(decision.span_attention_range, 4),
                                        "span_attention_progress": round(decision.span_attention_progress, 4),
                                        "left_anchor_probability_mean": round(decision.left_anchor_probability_mean, 4),
                                        "left_anchor_ink_support_mean": round(decision.left_anchor_ink_support_mean, 4),
                                        "left_anchor_attention_mean": round(decision.left_anchor_attention_mean, 4),
                                        "right_anchor_probability_mean": round(decision.right_anchor_probability_mean, 4),
                                        "right_anchor_ink_support_mean": round(decision.right_anchor_ink_support_mean, 4),
                                        "right_anchor_attention_mean": round(decision.right_anchor_attention_mean, 4),
                                        "span_visual_coverage_gain_mean": round(decision.span_visual_coverage_gain_mean, 4),
                                        "span_reused_attention_ratio_mean": round(decision.span_reused_attention_ratio_mean, 4),
                                        "global_span_start": round(decision.global_span_start, 4),
                                        "global_span_end": round(decision.global_span_end, 4),
                                        "probability_threshold": round(
                                            decision.probability_threshold, 4
                                        ),
                                        "ink_support_threshold": round(
                                            decision.ink_support_threshold, 4
                                        ),
                                        "tail_probability_mean": round(
                                            float(
                                                np.mean(
                                                    trace.token_probabilities[
                                                        -tail_count:
                                                    ]
                                                )
                                            ),
                                            4,
                                        )
                                        if tail_count
                                        else 0.0,
                                        "tail_ink_support_mean": round(
                                            float(
                                                np.mean(
                                                    trace.ink_support[-tail_count:]
                                                )
                                            ),
                                            4,
                                        )
                                        if tail_count
                                        else 0.0,
                                        "tail_attention_range": round(
                                            float(
                                                np.ptp(
                                                    trace.attention_centers[
                                                        -tail_count:
                                                    ]
                                                )
                                            ),
                                            4,
                                        )
                                        if tail_count
                                        else 0.0,
                                        "last_ink_position": round(
                                            trace.last_ink_position, 4
                                        ),
                                    }
                                )
                            if not applied:
                                continue
                            page_metrics[page][
                                "decoder_evidence_trimmed_count"
                            ] += 1
                            page_metrics[page][
                                "decoder_evidence_trimmed_char_count"
                            ] += len(decision.removed_text)
                            if decision.span_kind == "midline":
                                page_metrics[page][
                                    "decoder_evidence_midline_trimmed_count"
                                ] += 1
                            segment_texts[index] = decision.text
                            decoder_changed_indices.add(index)

                        if stop_page:
                            break
                    if stop_page:
                        break

        # Line-level visual grounding: use the next split as the right anchor.
        # This runs after local decisions so a stronger cross-segment decision may
        # safely widen a partial suffix trim, but never rewrites text.
        if (
            self.settings.decoder_evidence_enabled
            and self.settings.decoder_evidence_cross_segment_enabled
            and decoder_traces
        ):
            for original_index, segment_indices in enumerate(by_original):
                if len(segment_indices) < 2:
                    continue
                page = _page_index(crops[original_index].crop_id)
                page_metrics[page]["decoder_evidence_line_evidence_count"] += 1
                line_width = max(1, crops[original_index].image.width)
                for left_index, right_index in zip(segment_indices, segment_indices[1:]):
                    left_trace = decoder_traces.get(left_index)
                    right_trace = decoder_traces.get(right_index)
                    if left_trace is None or right_trace is None:
                        continue
                    page_metrics[page]["decoder_evidence_cross_segment_candidate_count"] += 1
                    cross_started = time.perf_counter()
                    decision = _cross_segment_suffix_decision(
                        left_trace,
                        right_trace,
                        segments[left_index],
                        segments[right_index],
                        line_width,
                        inference_images[left_index].width,
                        inference_images[right_index].width,
                        self.settings,
                    )
                    page_metrics[page]["cross_segment_analysis_ms"] += (
                        time.perf_counter() - cross_started
                    ) * 1000.0
                    if decision is None:
                        page_metrics[page]["decoder_evidence_cross_segment_rejected_count"] += 1
                        rejection_events = page_metrics[page][
                            "decoder_evidence_cross_segment_rejections"
                        ]
                        if (
                            len(rejection_events)
                            < self.settings.decoder_evidence_cross_segment_rejection_log_limit
                        ):
                            reasons = _cross_segment_rejection_reasons(
                                left_trace, right_trace, self.settings
                            )
                            rejection_events.append(
                                {
                                    "left_segment_index": left_index,
                                    "right_segment_index": right_index,
                                    "left_text": segment_texts[left_index],
                                    "right_text": segment_texts[right_index],
                                    "failed_conditions": list(reasons),
                                    "evidence": _cross_segment_rejection_diagnostics(
                                        left_trace, right_trace
                                    ),
                                }
                            )
                        continue

                    if decision.numeric_only_warning:
                        numeric_warning_segment_indices.add(left_index)
                        page_metrics[page]["decoder_evidence_suspicious_numeric_count"] += 1
                    secondary_relation = "disabled"
                    secondary_text = ""
                    secondary_confidence = 0.0
                    secondary_proposed_similarity = 0.0
                    secondary_raw_similarity = 0.0
                    if self.settings.secondary_recognizer_enabled:
                        raw_parts = [
                            decoder_traces.get(idx).raw_text
                            if decoder_traces.get(idx) is not None
                            else segment_texts[idx]
                            for idx in segment_indices
                        ]
                        proposed_parts = list(raw_parts)
                        proposed_parts[segment_indices.index(left_index)] = decision.text
                        raw_line, _ = _join_segments(
                            raw_parts,
                            max_token_overlap=self.settings.seam_max_token_overlap,
                            max_char_overlap=self.settings.seam_max_char_overlap,
                        )
                        proposed_line, _ = _join_segments(
                            proposed_parts,
                            max_token_overlap=self.settings.seam_max_token_overlap,
                            max_char_overlap=self.settings.seam_max_char_overlap,
                        )
                        verifier_started = time.perf_counter()
                        secondary = self._secondary_recognize_line(crops[original_index].image)
                        verifier_elapsed = (time.perf_counter() - verifier_started) * 1000
                        page_metrics[page]["secondary_verifier_count"] += 1
                        page_metrics[page]["secondary_verifier_ms"] += verifier_elapsed
                        if secondary is None:
                            page_metrics[page]["secondary_verifier_error_count"] += 1
                            secondary_relation = "unavailable"
                        else:
                            secondary_text = secondary.text
                            secondary_confidence = secondary.confidence
                            if secondary.confidence < self.settings.secondary_recognizer_min_confidence:
                                secondary_relation = "low_confidence"
                                page_metrics[page]["secondary_verifier_ambiguous_count"] += 1
                            else:
                                (
                                    secondary_relation,
                                    secondary_proposed_similarity,
                                    secondary_raw_similarity,
                                ) = _secondary_prefers_deletion(
                                    raw_line,
                                    proposed_line,
                                    secondary.text,
                                    self.settings.secondary_recognizer_preference_margin,
                                )
                                if secondary_relation == "primary_extra":
                                    page_metrics[page]["secondary_verifier_primary_extra_count"] += 1
                                elif secondary_relation == "conflict":
                                    page_metrics[page]["secondary_verifier_conflict_count"] += 1
                                else:
                                    page_metrics[page]["secondary_verifier_ambiguous_count"] += 1
                    applied = bool(
                        self.settings.decoder_evidence_apply_changes
                        and not decision.numeric_only_warning
                    )
                    if (
                        applied
                        and self.settings.secondary_recognizer_enabled
                        and self.settings.secondary_recognizer_apply_changes
                        and secondary_relation != "primary_extra"
                    ):
                        applied = False
                    events = page_metrics[page]["decoder_evidence_events"]
                    if len(events) < self.settings.decoder_evidence_event_limit:
                        events.append(
                            {
                                "segment_index": left_index,
                                "next_segment_index": right_index,
                                "risk": round(decoder_scores.get(left_index, 0.0), 3),
                                "raw_text": left_trace.raw_text,
                                "right_anchor_text": right_trace.raw_text,
                                "proposed_text": decision.text,
                                "validated_text": (
                                    decision.text if applied else segment_texts[left_index]
                                ),
                                "applied": applied,
                                "removed_text": decision.removed_text,
                                "reason": decision.reason,
                                "span_kind": decision.span_kind,
                                "numeric_warning_only": decision.numeric_only_warning,
                                "secondary_relation": secondary_relation,
                                "secondary_text": secondary_text,
                                "secondary_confidence": round(secondary_confidence, 4),
                                "secondary_proposed_similarity": round(secondary_proposed_similarity, 4),
                                "secondary_raw_similarity": round(secondary_raw_similarity, 4),
                                "span_start": decision.span_start,
                                "span_end": decision.span_end,
                                "span_probability_mean": round(decision.span_probability_mean, 4),
                                "span_ink_support_mean": round(decision.span_ink_support_mean, 4),
                                "span_visual_coverage_gain_mean": round(decision.span_visual_coverage_gain_mean, 4),
                                "span_reused_attention_ratio_mean": round(decision.span_reused_attention_ratio_mean, 4),
                                "left_anchor_probability_mean": round(decision.left_anchor_probability_mean, 4),
                                "left_anchor_ink_support_mean": round(decision.left_anchor_ink_support_mean, 4),
                                "left_anchor_visual_coverage_gain_mean": round(decision.left_anchor_visual_coverage_gain_mean, 4),
                                "right_anchor_probability_mean": round(decision.right_anchor_probability_mean, 4),
                                "right_anchor_ink_support_mean": round(decision.right_anchor_ink_support_mean, 4),
                                "right_anchor_visual_coverage_gain_mean": round(decision.right_anchor_visual_coverage_gain_mean, 4),
                                "global_span_start": round(decision.global_span_start, 4),
                                "global_span_end": round(decision.global_span_end, 4),
                                "right_anchor_global_attention_mean": round(decision.right_anchor_global_attention_mean, 4),
                            }
                        )
                    if not applied:
                        continue

                    previous_text = segment_texts[left_index]
                    newly_removed = max(0, len(previous_text) - len(decision.text))
                    already_changed = left_index in decoder_changed_indices
                    segment_texts[left_index] = decision.text
                    decoder_changed_indices.add(left_index)
                    page_metrics[page]["decoder_evidence_cross_segment_trimmed_count"] += 1
                    page_metrics[page]["decoder_evidence_visual_coverage_exhausted_count"] += 1
                    if decision.span_reused_attention_ratio_mean >= self.settings.decoder_evidence_cross_segment_min_reuse_ratio:
                        page_metrics[page]["decoder_evidence_attention_reuse_count"] += 1
                    if not already_changed:
                        page_metrics[page]["decoder_evidence_trimmed_count"] += 1
                    page_metrics[page]["decoder_evidence_trimmed_char_count"] += newly_removed

        # Multi-view consensus: nhận dạng lại các segment có rủi ro bằng crop
        # sát mực/dominant-ink. Chỉ xóa token original-only; không thay thế hoặc
        # bổ sung token, nên không phụ thuộc từ điển hay hard-code cụm tiếng Việt.
        guard_changed_indices: set[int] = set()
        guard_retry_ms_total = 0.0
        if (
            self.settings.hallucination_guard_enabled
            and self.settings.hallucination_guard_max_rechecks_per_page > 0
        ):
            guard_candidates: dict[int, list[tuple[float, int, _RetryView]]] = {}
            for index, segment in enumerate(segments):
                retry_view = _build_retry_view(segment.image, self.settings)
                risk = _hallucination_risk(
                    segment,
                    segment_texts[index],
                    segment_probabilities[index],
                    retry_view,
                    self.settings,
                )
                if (
                    retry_view is not None
                    and risk >= self.settings.hallucination_guard_risk_threshold
                ):
                    guard_candidates.setdefault(segment.page_index, []).append(
                        (risk, index, retry_view)
                    )

            for page, values in guard_candidates.items():
                values.sort(key=lambda item: item[0], reverse=True)
                page_metrics[page]["hallucination_guard_candidate_count"] = len(values)
                for risk, index, retry_view in values[
                    : self.settings.hallucination_guard_max_rechecks_per_page
                ]:
                    original_text = segment_texts[index]
                    original_probability = segment_probabilities[index]
                    started = time.perf_counter()
                    retry = self._recognize_one(predictor, retry_view.image)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    guard_retry_ms_total += elapsed_ms
                    page_metrics[page]["hallucination_guard_retry_count"] += 1
                    page_metrics[page]["hallucination_guard_ms"] += elapsed_ms
                    if retry_view.kind == "dominant_ink":
                        page_metrics[page][
                            "hallucination_guard_dominant_ink_retry_count"
                        ] += 1
                    if not retry.text or retry.text == original_text:
                        continue
                    page_metrics[page]["hallucination_guard_disagreement_count"] += 1
                    decision = _consensus_delete_only(
                        original_text,
                        retry.text,
                        original_confidence=original_probability,
                        retry_confidence=retry.confidence,
                        leading_blank_ratio=retry_view.leading_blank_ratio,
                        trailing_blank_ratio=retry_view.trailing_blank_ratio,
                        width_reduction_ratio=retry_view.width_reduction_ratio,
                        settings=self.settings,
                    )
                    if decision is None:
                        continue
                    if decision.numeric_only_warning:
                        numeric_warning_segment_indices.add(index)
                        page_metrics[page][
                            "hallucination_guard_numeric_removed_count"
                        ] += 1
                        continue
                    page_metrics[page]["hallucination_guard_consensus_count"] += 1
                    page_metrics[page]["hallucination_guard_removed_token_count"] += len(
                        decision.removed_tokens
                    )
                    page_metrics[page][
                        f"hallucination_guard_{decision.mode}_removed_count"
                    ] += 1
                    if any(
                        any(char.isdigit() for char in token)
                        for token in decision.removed_tokens
                    ):
                        page_metrics[page]["hallucination_guard_numeric_removed_count"] += 1
                    events = page_metrics[page]["hallucination_guard_events"]
                    if len(events) < self.settings.hallucination_guard_log_event_limit:
                        events.append(
                            {
                                "segment_index": index,
                                "view": retry_view.kind,
                                "risk": round(risk, 3),
                                "raw_text": original_text,
                                "retry_text": retry.text,
                                "validated_text": decision.text,
                                "removed_tokens": list(decision.removed_tokens),
                                "mode": decision.mode,
                            }
                        )
                    if self.settings.hallucination_guard_apply_changes:
                        segment_texts[index] = decision.text
                        segment_probabilities[index] = min(
                            original_probability, retry.confidence
                        )
                        guard_changed_indices.add(index)

        # Retry có mục tiêu cho tail segment có nhiều blank phải hoặc confidence
        # thấp. Crop retry được trim tới vùng mực, không liên quan padding batch.
        tail_uncertain_indices: set[int] = set()
        tail_suppressed_indices: set[int] = set()
        tail_retry_ms_total = 0.0
        if self.settings.tail_segment_validation:
            tail_candidates: dict[int, list[tuple[float, float, int]]] = {}
            for index, segment in enumerate(segments):
                if not (segment.was_split and segment.is_tail and segment_texts[index]):
                    continue
                trailing_blank = _trailing_blank_ratio(segment.image)
                low_confidence = (
                    segment_probabilities[index] < self.settings.tail_segment_min_confidence
                )
                low_ink = segment.ink_ratio < self.settings.tail_segment_low_ink_ratio
                suspicious = (
                    trailing_blank >= self.settings.tail_segment_trailing_blank_ratio
                    or (low_confidence and low_ink)
                )
                if suspicious:
                    tail_candidates.setdefault(segment.page_index, []).append(
                        (trailing_blank, -segment_probabilities[index], index)
                    )

            for page, values in tail_candidates.items():
                values.sort(reverse=True)
                retry_indices = (
                    {
                        item[2]
                        for item in values[
                            : self.settings.tail_segment_max_retries_per_page
                        ]
                    }
                    if self.settings.tail_segment_retry_enabled
                    else set()
                )
                for _, _, index in values:
                    segment = segments[index]
                    original_text = segment_texts[index]
                    original_probability = segment_probabilities[index]
                    original_numeric_tokens = _numeric_tokens(original_text)
                    if original_numeric_tokens:
                        numeric_warning_segment_indices.add(index)
                    if index in retry_indices:
                        retry_image = _trim_image_to_ink(segment.image)
                        if retry_image.size != segment.image.size:
                            started = time.perf_counter()
                            retry = self._recognize_one(predictor, retry_image)
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            tail_retry_ms_total += elapsed_ms
                            page_metrics[page]["tail_segment_retry_count"] += 1
                            page_metrics[page]["tail_segment_retry_ms"] += elapsed_ms
                            retry_clean = (
                                bool(retry.text)
                                and _find_decoder_loop(retry.text) is None
                                and _find_character_loop(retry.text) is None
                            )
                            original_tokens = len(_TOKEN.findall(original_text))
                            retry_tokens = len(_TOKEN.findall(retry.text))
                            keeps_enough = retry_tokens >= max(1, math.ceil(original_tokens * 0.45))
                            confidence_ok = retry.confidence >= original_probability - 0.05
                            likely_removed_tail = (
                                len(retry.text) <= len(original_text)
                                and retry.confidence >= original_probability - 0.02
                            )
                            retry_numeric_tokens = _numeric_tokens(retry.text)
                            numeric_content_changed = bool(
                                (original_numeric_tokens or retry_numeric_tokens)
                                and original_numeric_tokens != retry_numeric_tokens
                            )
                            if numeric_content_changed:
                                numeric_warning_segment_indices.add(index)
                            if (
                                not numeric_content_changed
                                and retry_clean
                                and keeps_enough
                                and (confidence_ok or likely_removed_tail)
                            ):
                                segment_texts[index] = retry.text
                                segment_probabilities[index] = retry.confidence
                                page_metrics[page]["tail_segment_retry_accepted_count"] += 1

                    current_text = segment_texts[index]
                    current_probability = segment_probabilities[index]
                    trailing_blank = _trailing_blank_ratio(segment.image)
                    no_visible_ink = segment.ink_columns_ratio < 0.01
                    very_low_ink = (
                        segment.ink_ratio < self.settings.tail_segment_low_ink_ratio * 0.35
                        and segment.ink_columns_ratio < 0.05
                    )
                    strong_hallucination = (
                        (no_visible_ink and bool(current_text))
                        or (
                            very_low_ink
                            and current_probability < 0.45
                            and len(current_text) >= 6
                        )
                    )
                    if (
                        self.settings.suppress_low_ink_tail_hallucinations
                        and strong_hallucination
                        and not _numeric_tokens(current_text)
                    ):
                        segment_texts[index] = ""
                        tail_suppressed_indices.add(index)
                        page_metrics[page]["tail_segment_suppressed_count"] += 1
                    elif (
                        current_text
                        and (
                            current_probability < self.settings.tail_segment_min_confidence
                            or trailing_blank >= self.settings.tail_segment_trailing_blank_ratio
                        )
                    ):
                        tail_uncertain_indices.add(index)
                        page_metrics[page]["tail_segment_uncertain_count"] += 1

        # Xử lý chuỗi ký tự/dấu câu lặp dài.
        char_loop_detected_count = 0
        char_loop_trimmed_count = 0
        trimmed_segment_indices: set[int] = set()
        for index, text in enumerate(segment_texts):
            start_char = _find_character_loop(text)
            if start_char is None:
                continue
            page = segments[index].page_index
            char_loop_detected_count += 1
            page_metrics[page]["decoder_char_loop_detected_count"] += 1
            if any(character.isdigit() for character in text[start_char:]):
                numeric_warning_segment_indices.add(index)
                continue
            trimmed = _trim_character_loop(text, start_char)
            if trimmed != text:
                segment_texts[index] = trimmed
                trimmed_segment_indices.add(index)
                char_loop_trimmed_count += 1
                page_metrics[page]["decoder_char_loop_trimmed_count"] += 1

        # Retry chỉ các loop token/n-gram thật sự, giới hạn theo trang.
        candidates: dict[int, list[tuple[tuple[int, int, int], float, int, _LoopIssue]]] = {}
        partial_loop_detected_count = 0
        for index, text in enumerate(segment_texts):
            issue = _find_decoder_loop(text)
            if issue is None:
                continue
            page = segments[index].page_index
            page_metrics[page]["decoder_loop_detected_count"] += 1
            if _decoder_loop_removed_contains_numeric(text, issue):
                numeric_warning_segment_indices.add(index)
                continue
            if issue.partial_tokens:
                partial_loop_detected_count += 1
                page_metrics[page]["decoder_partial_loop_detected_count"] += 1
            candidates.setdefault(page, []).append(
                (issue.severity, segment_probabilities[index], index, issue)
            )

        beam_retry_count = 0
        beam_retry_ms = 0.0
        trimmed_count = 0
        for page, values in candidates.items():
            values.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            retry_indices = {
                item[2]
                for item in values[: self.settings.max_beam_retries_per_page]
                if self.settings.beam_retry_enabled
            }
            for _, _, index, issue in values:
                original = segment_texts[index]
                if index not in retry_indices:
                    trimmed = _trim_decoder_loop(original, issue)
                    if trimmed != original:
                        segment_texts[index] = trimmed
                        trimmed_segment_indices.add(index)
                        trimmed_count += 1
                        page_metrics[page]["decoder_loop_trimmed_count"] += 1
                    continue
                started = time.perf_counter()
                retry_text = self._beam_retry(predictor, segments[index].image)
                elapsed_ms = (time.perf_counter() - started) * 1000
                beam_retry_ms += elapsed_ms
                beam_retry_count += 1
                page_metrics[page]["beam_retry_count"] += 1
                page_metrics[page]["beam_retry_ms"] += elapsed_ms
                retry_issue = _find_decoder_loop(retry_text)
                retry_char_issue = _find_character_loop(retry_text)
                minimum = math.ceil(len(original) * 0.45)
                if (
                    retry_text
                    and retry_issue is None
                    and retry_char_issue is None
                    and len(retry_text) >= minimum
                ):
                    segment_texts[index] = retry_text
                else:
                    trimmed = _trim_decoder_loop(original, issue)
                    if trimmed != original:
                        segment_texts[index] = trimmed
                        trimmed_segment_indices.add(index)
                        trimmed_count += 1
                        page_metrics[page]["decoder_loop_trimmed_count"] += 1

        for index, text in enumerate(segment_texts):
            if not text:
                page_metrics[segments[index].page_index]["empty_recognition_count"] += 1

        results: list[Recognition] = []
        seam_overlap_merge_count = 0
        for original_index, segment_indices in enumerate(by_original):
            texts = [segment_texts[index] for index in segment_indices]
            probabilities = [
                segment_probabilities[index]
                for index in segment_indices
                if segment_texts[index]
            ]
            text, seam_count = _join_segments(
                texts,
                max_token_overlap=self.settings.seam_max_token_overlap,
                max_char_overlap=self.settings.seam_max_char_overlap,
            )
            seam_overlap_merge_count += seam_count
            page = _page_index(crops[original_index].crop_id)
            page_metrics[page]["seam_overlap_merge_count"] += seam_count
            confidence = min(probabilities, default=0.0)
            if not text:
                results.append(self._failure())
                continue
            if any(
                index in numeric_warning_segment_indices
                for index in segment_indices
            ):
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="numeric_ocr_pattern_unresolved",
                        message_vi=(
                            "Phát hiện bất đồng hoặc mẫu lặp có chữ số; "
                            "giữ nguyên nội dung và yêu cầu đối chiếu PDF gốc"
                        ),
                    )
                )
                continue
            if any(index in decoder_changed_indices for index in segment_indices):
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="decoder_attention_span_trimmed",
                        message_vi=(
                            "Đã loại một span không còn đủ bằng chứng attention/pixel "
                            "từ ảnh; cần đối chiếu PDF gốc"
                        ),
                    )
                )
                continue
            if any(index in trimmed_segment_indices for index in segment_indices):
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="decoder_loop_trimmed",
                        message_vi=(
                            "Đã cắt phần lặp bất thường của dòng; cần đối chiếu PDF gốc"
                        ),
                    )
                )
                continue
            if any(index in guard_changed_indices for index in segment_indices):
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="unsupported_ocr_insertion_removed",
                        message_vi=(
                            "Đã loại token không được pass crop sát mực xác nhận; "
                            "cần đối chiếu PDF gốc"
                        ),
                    )
                )
                continue
            if any(index in tail_suppressed_indices for index in segment_indices):
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="tail_segment_suppressed",
                        message_vi=(
                            "Đã bỏ segment cuối có rất ít mực nhưng sinh text bất thường; "
                            "cần đối chiếu PDF gốc"
                        ),
                    )
                )
                continue
            if any(index in tail_uncertain_indices for index in segment_indices):
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="tail_segment_uncertain",
                        message_vi=(
                            "Segment cuối của dòng có blank lớn hoặc confidence thấp; "
                            "cần đối chiếu PDF gốc"
                        ),
                    )
                )
                continue
            if original_index in width_cap_unresolved:
                results.append(
                    Recognition(
                        text=text,
                        confidence=max(0.0, min(1.0, confidence)),
                        error_code="width_cap_unresolved",
                        message_vi=(
                            "Dòng vượt giới hạn chiều rộng VietOCR và không tìm được "
                            "khoảng trắng an toàn để chia"
                        ),
                    )
                )
                continue
            results.append(self._success(text, confidence))

        if not self.settings.semantic_selective_verification_enabled:
            results = self._apply_semantic_verification(crops, results, page_metrics)
        else:
            results = [
                (
                    Recognition(
                        text=recognition.text,
                        confidence=recognition.confidence,
                        error_code=recognition.error_code,
                        message_vi=recognition.message_vi,
                        raw_text=recognition.raw_text,
                        semantic_risk="high",
                        semantic_reasons=tuple(
                            dict.fromkeys(
                                (*recognition.semantic_reasons, "primary_recognition_risk")
                            )
                        ),
                        secondary_confidence=recognition.secondary_confidence,
                        verifier_text=recognition.verifier_text,
                        verifier_confidence=recognition.verifier_confidence,
                    )
                    if recognition.semantic_risk == "none"
                    and (
                        recognition.error_code is not None
                        or recognition.confidence
                        < self.settings.semantic_primary_low_confidence
                    )
                    else recognition
                )
                for recognition in results
            ]
            for page_index, metrics in page_metrics.items():
                metrics["semantic_deferred_count"] = sum(
                    _page_index(crop.crop_id) == page_index for crop in crops
                )
                metrics.setdefault("semantic_verification_ms", 0.0)

        for metrics in page_metrics.values():
            metrics["greedy_batch_ms_allocated"] = round(
                metrics["greedy_batch_ms_allocated"], 3
            )
            metrics["beam_retry_ms"] = round(metrics["beam_retry_ms"], 3)
            metrics["tail_segment_retry_ms"] = round(
                metrics["tail_segment_retry_ms"], 3
            )
            metrics["hallucination_guard_ms"] = round(
                metrics["hallucination_guard_ms"], 3
            )
            metrics["decoder_evidence_ms"] = round(
                metrics["decoder_evidence_ms"], 3
            )
            for diagnostic_key in (
                "greedy_model_cpu_ms_allocated",
                "decoder_preprocess_wall_ms", "decoder_preprocess_cpu_ms",
                "decoder_encoder_wall_ms", "decoder_encoder_cpu_ms",
                "decoder_model_wall_ms", "decoder_model_cpu_ms",
                "decoder_attention_extract_wall_ms", "decoder_attention_extract_cpu_ms",
                "decoder_torch_postprocess_wall_ms", "decoder_torch_postprocess_cpu_ms",
                "decoder_visual_grounding_wall_ms", "decoder_visual_grounding_cpu_ms",
                "decoder_trace_build_wall_ms", "decoder_trace_build_cpu_ms",
                "cross_segment_analysis_ms",
            ):
                metrics[diagnostic_key] = round(float(metrics.get(diagnostic_key, 0.0)), 3)
            metrics["secondary_verifier_ms"] = round(
                metrics["secondary_verifier_ms"], 3
            )
            metrics["semantic_verification_ms"] = round(
                float(metrics.get("semantic_verification_ms", 0.0)), 3
            )
            metrics["recognition_ms_allocated"] = round(
                metrics["greedy_batch_ms_allocated"]
                + metrics["beam_retry_ms"]
                + metrics["tail_segment_retry_ms"]
                + metrics["hallucination_guard_ms"]
                + metrics["decoder_evidence_ms"]
                + metrics["cross_segment_analysis_ms"]
                + metrics["secondary_verifier_ms"]
                + metrics["semantic_verification_ms"],
                3,
            )

        recognizer_wall_ms = (time.perf_counter() - recognize_wall_started) * 1000.0
        recognizer_cpu_ms = (time.process_time() - recognize_cpu_started) * 1000.0
        if self.settings.performance_diagnostics_enabled:
            diagnostics_started = time.perf_counter()
            runtime_end = _runtime_snapshot()
            diagnostics_overhead_ms += (time.perf_counter() - diagnostics_started) * 1000.0
        else:
            runtime_end = {}
        workload = {
            "segment_count": len(segments),
            "greedy_batch_count": batch_count,
            "greedy_input_pixel_count": sum(image.width * image.height for image in inference_images),
            "trace_count": sum(m["decoder_evidence_trace_count"] for m in page_metrics.values()),
            "trace_batch_count": sum(m["decoder_evidence_trace_batch_count"] for m in page_metrics.values()),
            "decoder_forward_call_count": sum(int(m.get("decoder_forward_call_count", 0)) for m in page_metrics.values()),
            "decoder_sample_step_count": sum(int(m.get("decoder_sample_step_count", 0)) for m in page_metrics.values()),
            "decoder_attention_element_count": sum(int(m.get("decoder_attention_element_count", 0)) for m in page_metrics.values()),
            "trace_input_pixel_count": sum(int(m.get("decoder_trace_input_pixel_count", 0)) for m in page_metrics.values()),
            "trace_character_count": sum(int(m.get("decoder_trace_character_count", 0)) for m in page_metrics.values()),
        }
        workload["fingerprint"] = _workload_hash(workload)
        accounted_ms = (
            greedy_ms
            + decoder_evidence_ms_total
            + guard_retry_ms_total
            + tail_retry_ms_total
            + sum(float(m.get("beam_retry_ms", 0.0)) for m in page_metrics.values())
            + sum(float(m.get("cross_segment_analysis_ms", 0.0)) for m in page_metrics.values())
            + sum(float(m.get("secondary_verifier_ms", 0.0)) for m in page_metrics.values())
            + sum(float(m.get("semantic_verification_ms", 0.0)) for m in page_metrics.values())
        )
        performance_diagnostics = {
            "recognizer_wall_ms": round(recognizer_wall_ms, 3),
            "recognizer_cpu_ms": round(recognizer_cpu_ms, 3),
            "cpu_wall_ratio": round(recognizer_cpu_ms / max(recognizer_wall_ms, 1e-9), 4),
            "accounted_wall_ms": round(accounted_ms + predictor_resolve_wall_ms, 3),
            "unaccounted_wall_ms": round(max(0.0, recognizer_wall_ms - accounted_ms - predictor_resolve_wall_ms), 3),
            "diagnostics_overhead_ms": round(diagnostics_overhead_ms, 3),
            "predictor_resolve_wall_ms": round(predictor_resolve_wall_ms, 3),
            "predictor_resolve_cpu_ms": round(predictor_resolve_cpu_ms, 3),
            "greedy_model_wall_ms": round(greedy_ms, 3),
            "greedy_model_cpu_ms": round(greedy_cpu_ms, 3),
            "decoder_evidence_wall_ms": round(decoder_evidence_ms_total, 3),
            "cross_segment_analysis_ms": round(sum(float(m.get("cross_segment_analysis_ms", 0.0)) for m in page_metrics.values()), 3),
            "runtime_start": runtime_start,
            "runtime_end": runtime_end,
            "visual_grounding_enabled": self.settings.decoder_evidence_visual_grounding_enabled,
            "cross_segment_enabled": self.settings.decoder_evidence_cross_segment_enabled,
        }
        for metrics in page_metrics.values():
            metrics["performance_diagnostics"] = performance_diagnostics
            metrics["workload_fingerprint"] = workload

        self.last_page_metrics = page_metrics
        self.last_batch_metrics = {
            "crop_count": len(crops),
            **split_metrics,
            "page_count": len(page_metrics),
            "batch_count": batch_count,
            "mean_batch_size": round(len(segments) / max(batch_count, 1), 2),
            "max_batch_size": max_batch_size,
            "effective_batch_size": round(len(segments) / max(batch_count, 1), 2),
            "batch_sizes": batch_sizes,
            "normalized_batch_widths": normalized_batch_widths,
            "padded_segment_count": padded_segment_count,
            "padding_source_columns": padding_source_columns,
            "internal_batch_count_estimate": batch_count,
            "greedy_batch_ms": round(greedy_ms, 3),
            "greedy_model_cpu_ms": round(greedy_cpu_ms, 3),
            "greedy_batch_fallback_count": sum(
                metrics["greedy_batch_fallback_count"] for metrics in page_metrics.values()
            ),
            "greedy_batch_fallback_segment_count": sum(
                metrics["greedy_batch_fallback_segment_count"] for metrics in page_metrics.values()
            ),
            "greedy_batch_fallback_error_types": {
                error_type: sum(
                    metrics["greedy_batch_fallback_error_types"].get(error_type, 0)
                    for metrics in page_metrics.values()
                )
                for error_type in sorted(
                    {
                        error_type
                        for metrics in page_metrics.values()
                        for error_type in metrics["greedy_batch_fallback_error_types"]
                    }
                )
            },
            "performance_diagnostics": performance_diagnostics,
            "workload_fingerprint": workload,
            "tail_segment_retry_count": sum(
                metrics["tail_segment_retry_count"] for metrics in page_metrics.values()
            ),
            "tail_segment_retry_accepted_count": sum(
                metrics["tail_segment_retry_accepted_count"]
                for metrics in page_metrics.values()
            ),
            "tail_segment_uncertain_count": sum(
                metrics["tail_segment_uncertain_count"] for metrics in page_metrics.values()
            ),
            "tail_segment_suppressed_count": sum(
                metrics["tail_segment_suppressed_count"] for metrics in page_metrics.values()
            ),
            "tail_segment_retry_ms": round(tail_retry_ms_total, 3),
            "hallucination_guard_candidate_count": sum(
                metrics["hallucination_guard_candidate_count"]
                for metrics in page_metrics.values()
            ),
            "hallucination_guard_retry_count": sum(
                metrics["hallucination_guard_retry_count"]
                for metrics in page_metrics.values()
            ),
            "hallucination_guard_disagreement_count": sum(
                metrics["hallucination_guard_disagreement_count"]
                for metrics in page_metrics.values()
            ),
            "hallucination_guard_consensus_count": sum(
                metrics["hallucination_guard_consensus_count"]
                for metrics in page_metrics.values()
            ),
            "hallucination_guard_removed_token_count": sum(
                metrics["hallucination_guard_removed_token_count"]
                for metrics in page_metrics.values()
            ),
            "hallucination_guard_numeric_removed_count": sum(
                metrics["hallucination_guard_numeric_removed_count"]
                for metrics in page_metrics.values()
            ),
            "hallucination_guard_ms": round(guard_retry_ms_total, 3),
            "decoder_evidence_candidate_count": sum(
                metrics["decoder_evidence_candidate_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_seed_selected_count": sum(
                metrics["decoder_evidence_seed_selected_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_context_forced_count": sum(
                metrics["decoder_evidence_context_forced_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_selected_count": sum(
                metrics["decoder_evidence_selected_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_unchecked_candidate_count": sum(
                metrics["decoder_evidence_unchecked_candidate_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trace_count": sum(
                metrics["decoder_evidence_trace_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trace_batch_count": sum(
                metrics["decoder_evidence_trace_batch_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trace_batch_size_max": max(
                (metrics["decoder_evidence_trace_batch_size_max"] for metrics in page_metrics.values()),
                default=0,
            ),
            "decoder_evidence_supported_count": sum(
                metrics["decoder_evidence_supported_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trace_mismatch_count": sum(
                metrics["decoder_evidence_trace_mismatch_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trace_error_count": sum(
                metrics["decoder_evidence_trace_error_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trace_error_types": {
                error_type: sum(
                    metrics["decoder_evidence_trace_error_types"].get(error_type, 0)
                    for metrics in page_metrics.values()
                )
                for error_type in sorted(
                    {
                        error_type
                        for metrics in page_metrics.values()
                        for error_type in metrics[
                            "decoder_evidence_trace_error_types"
                        ]
                    }
                )
            },
            "decoder_evidence_disabled_reason": decoder_evidence_disabled_reason,
            "decoder_evidence_circuit_breaker_count": sum(
                metrics["decoder_evidence_circuit_breaker_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_attention_stall_count": sum(
                metrics["decoder_evidence_attention_stall_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_visual_exhausted_count": sum(
                metrics["decoder_evidence_visual_exhausted_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_near_loop_count": sum(
                metrics["decoder_evidence_near_loop_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_midline_span_count": sum(
                metrics["decoder_evidence_midline_span_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_midline_trimmed_count": sum(
                metrics["decoder_evidence_midline_trimmed_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_line_evidence_count": sum(
                metrics["decoder_evidence_line_evidence_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_cross_segment_candidate_count": sum(
                metrics["decoder_evidence_cross_segment_candidate_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_cross_segment_trimmed_count": sum(
                metrics["decoder_evidence_cross_segment_trimmed_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_cross_segment_rejected_count": sum(
                metrics["decoder_evidence_cross_segment_rejected_count"]
                for metrics in page_metrics.values()
            ),
            "secondary_verifier_count": sum(
                metrics["secondary_verifier_count"] for metrics in page_metrics.values()
            ),
            "secondary_verifier_primary_extra_count": sum(
                metrics["secondary_verifier_primary_extra_count"]
                for metrics in page_metrics.values()
            ),
            "secondary_verifier_conflict_count": sum(
                metrics["secondary_verifier_conflict_count"]
                for metrics in page_metrics.values()
            ),
            "secondary_verifier_ambiguous_count": sum(
                metrics["secondary_verifier_ambiguous_count"]
                for metrics in page_metrics.values()
            ),
            "secondary_verifier_error_count": sum(
                metrics["secondary_verifier_error_count"]
                for metrics in page_metrics.values()
            ),
            "secondary_verifier_ms": round(
                sum(metrics["secondary_verifier_ms"] for metrics in page_metrics.values()), 3
            ),
            "semantic_verified_count": sum(
                int(metrics.get("semantic_verified_count", 0))
                for metrics in page_metrics.values()
            ),
            "semantic_deferred_count": sum(
                int(metrics.get("semantic_deferred_count", 0))
                for metrics in page_metrics.values()
            ),
            "semantic_secondary_unavailable_count": sum(
                int(metrics.get("semantic_secondary_unavailable_count", 0))
                for metrics in page_metrics.values()
            ),
            "semantic_secondary_error_count": sum(
                int(metrics.get("semantic_secondary_error_count", 0))
                for metrics in page_metrics.values()
            ),
            "semantic_auto_trimmed_count": sum(
                int(metrics.get("semantic_auto_trimmed_count", 0))
                for metrics in page_metrics.values()
            ),
            "semantic_high_risk_count": sum(
                int(metrics.get("semantic_high_risk_count", 0))
                for metrics in page_metrics.values()
            ),
            "semantic_verification_ms": round(
                sum(
                    float(metrics.get("semantic_verification_ms", 0.0))
                    for metrics in page_metrics.values()
                ),
                3,
            ),
            "decoder_evidence_visual_coverage_exhausted_count": sum(
                metrics["decoder_evidence_visual_coverage_exhausted_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_attention_reuse_count": sum(
                metrics["decoder_evidence_attention_reuse_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trimmed_count": sum(
                metrics["decoder_evidence_trimmed_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_trimmed_char_count": sum(
                metrics["decoder_evidence_trimmed_char_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_suspicious_numeric_count": sum(
                metrics["decoder_evidence_suspicious_numeric_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_cluster_expansion_count": sum(
                metrics["decoder_evidence_cluster_expansion_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_expanded_word_count": sum(
                metrics["decoder_evidence_expanded_word_count"]
                for metrics in page_metrics.values()
            ),
            "decoder_evidence_ms": round(decoder_evidence_ms_total, 3),
            "seam_overlap_merge_count": seam_overlap_merge_count,
            "beam_retry_count": beam_retry_count,
            "beam_retry_ms": round(beam_retry_ms, 3),
            "decoder_loop_detected_count": sum(len(values) for values in candidates.values()),
            "decoder_partial_loop_detected_count": partial_loop_detected_count,
            "decoder_loop_trimmed_count": trimmed_count,
            "decoder_char_loop_detected_count": char_loop_detected_count,
            "decoder_char_loop_trimmed_count": char_loop_trimmed_count,
            "empty_recognition_count": sum(
                metrics["empty_recognition_count"] for metrics in page_metrics.values()
            ),
        }
        return results

    def _beam_retry(self, predictor: Any, image: Image.Image) -> str:
        config = getattr(predictor, "config", None)
        predictor_config = config.get("predictor") if isinstance(config, dict) else None
        if not isinstance(predictor_config, dict) or not hasattr(predictor, "predict"):
            return ""
        previous = predictor_config.get("beamsearch", False)
        try:
            predictor_config["beamsearch"] = True
            with _inference_context(self.settings.use_torch_inference_mode):
                value = predictor.predict(image, return_prob=False)
            if isinstance(value, tuple):
                value = value[0]
            return normalize_vietnamese_text(str(value))
        except Exception:
            return ""
        finally:
            predictor_config["beamsearch"] = previous

    def _recognize_one(self, predictor: Any, image: Image.Image) -> Recognition:
        try:
            with _inference_context(self.settings.use_torch_inference_mode):
                value = predictor.predict(image, return_prob=True)
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("VietOCR trả kết quả không đúng định dạng")
            return self._success(str(value[0]), float(value[1]))
        except Exception:
            return self._failure()

    @staticmethod
    def _success(text: str, probability: float) -> Recognition:
        return Recognition(
            text=normalize_vietnamese_text(text),
            confidence=max(0.0, min(1.0, probability)),
        )

    @staticmethod
    def _failure() -> Recognition:
        return Recognition(
            text="",
            confidence=0.0,
            error_code="recognition_failed",
            message_vi="Không nhận dạng được dòng chữ",
        )
