from __future__ import annotations

import csv
import io
import re
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Sequence

from .config import Settings
from .models import BBox, PageImage, Recognition


_LEXICAL_TOKEN = re.compile(r"[\w]+(?:[/.-][\w]+)*", re.UNICODE)


@dataclass(frozen=True)
class TesseractLine:
    text: str
    confidence: float
    bbox: BBox


def _strict_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFC", match.group(0).casefold())
        for match in _LEXICAL_TOKEN.finditer(value)
    )


def _accentless(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def _accentless_material(value: str) -> str:
    return " ".join(_accentless(token) for token in _strict_tokens(value))


def parse_tesseract_tsv(value: str) -> list[TesseractLine]:
    """Parse word-level Tesseract TSV without treating OCR quotes as CSV syntax."""
    reader = csv.DictReader(
        io.StringIO(value),
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    )
    grouped: dict[tuple[int, int, int, int], list[tuple[str, float, BBox]]] = {}
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf") or -1.0)
            if confidence < 0.0:
                continue
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
            key = (
                int(row["page_num"]),
                int(row["block_num"]),
                int(row["par_num"]),
                int(row["line_num"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(
            (
                text,
                confidence / 100.0,
                BBox(left, top, left + width, top + height),
            )
        )

    lines: list[TesseractLine] = []
    for words in grouped.values():
        text = " ".join(word[0] for word in words)
        weights = [max(len(word[0]), 1) for word in words]
        confidence = sum(
            word[1] * weight for word, weight in zip(words, weights)
        ) / sum(weights)
        boxes = [word[2] for word in words]
        lines.append(
            TesseractLine(
                text=text,
                confidence=max(0.0, min(1.0, confidence)),
                bbox=BBox(
                    min(box.x0 for box in boxes),
                    min(box.y0 for box in boxes),
                    max(box.x1 for box in boxes),
                    max(box.y1 for box in boxes),
                ),
            )
        )
    return sorted(lines, key=lambda line: (line.bbox.y0, line.bbox.x0))


def _line_match_score(left: BBox, right: BBox) -> float:
    vertical_intersection = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    horizontal_intersection = max(
        0.0,
        min(left.x1, right.x1) - max(left.x0, right.x0),
    )
    vertical_overlap = vertical_intersection / max(min(left.height, right.height), 1.0)
    horizontal_overlap = horizontal_intersection / max(min(left.width, right.width), 1.0)
    return 0.75 * min(vertical_overlap, 1.0) + 0.25 * min(horizontal_overlap, 1.0)


def match_tesseract_lines(
    crop_boxes: Sequence[BBox],
    lines: Sequence[TesseractLine],
    *,
    min_score: float,
) -> list[TesseractLine | None]:
    """Greedily create one-to-one matches using page coordinates."""
    candidates = sorted(
        (
            (_line_match_score(crop_box, line.bbox), crop_index, line_index)
            for crop_index, crop_box in enumerate(crop_boxes)
            for line_index, line in enumerate(lines)
        ),
        reverse=True,
    )
    matches: list[TesseractLine | None] = [None] * len(crop_boxes)
    used_lines: set[int] = set()
    for score, crop_index, line_index in candidates:
        if score < min_score:
            break
        if matches[crop_index] is not None or line_index in used_lines:
            continue
        matches[crop_index] = lines[line_index]
        used_lines.add(line_index)
    return matches


def apply_tesseract_verification(
    primary: Recognition,
    verifier: TesseractLine | None,
    settings: Settings,
) -> Recognition:
    """Attach independent evidence and fail closed; never rewrite OCR text."""
    if verifier is None:
        return primary
    result = replace(
        primary,
        verifier_text=verifier.text,
        verifier_confidence=verifier.confidence,
    )
    if verifier.confidence < settings.tesseract_min_confidence:
        return result

    primary_tokens = _strict_tokens(primary.text)
    verifier_tokens = _strict_tokens(verifier.text)
    if not primary_tokens or not verifier_tokens or primary_tokens == verifier_tokens:
        return result

    primary_numbers = tuple(
        token for token in primary_tokens if any(character.isdigit() for character in token)
    )
    verifier_numbers = tuple(
        token for token in verifier_tokens if any(character.isdigit() for character in token)
    )
    primary_material = _accentless_material(primary.text)
    verifier_material = _accentless_material(verifier.text)
    material_similarity = SequenceMatcher(
        None,
        primary_material,
        verifier_material,
    ).ratio()
    reason: str | None = None
    if primary_numbers != verifier_numbers and material_similarity >= 0.45:
        reason = "tesseract_numeric_disagreement"
    elif (
        tuple(_accentless(token) for token in primary_tokens)
        == tuple(_accentless(token) for token in verifier_tokens)
    ):
        reason = "tesseract_diacritic_disagreement"
    elif material_similarity >= settings.tesseract_material_similarity:
        reason = "tesseract_material_disagreement"

    if reason is None:
        return result
    return replace(
        result,
        semantic_risk="high",
        semantic_reasons=tuple(dict.fromkeys((*primary.semantic_reasons, reason))),
    )


def _mark_high_risk(primary: Recognition, reason: str) -> Recognition:
    return replace(
        primary,
        semantic_risk="high",
        semantic_reasons=tuple(dict.fromkeys((*primary.semantic_reasons, reason))),
    )


class TesseractPageVerifier:
    """Offline page-level Tesseract verifier for legal-text safety gates."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.executable = settings.resolve_project_path(
            settings.tesseract_executable_path
        )
        self.tessdata = settings.resolve_project_path(settings.tesseract_data_path)

    def _runtime_available(self) -> bool:
        languages = [
            language.strip()
            for language in self.settings.tesseract_languages.split("+")
            if language.strip()
        ]
        return bool(
            self.executable.is_file()
            and self.tessdata.is_dir()
            and all(
                (self.tessdata / f"{language}.traineddata").is_file()
                for language in languages
            )
        )

    def _run_image_tsv(self, image: Any, *, psm: int) -> str:
        image_bytes = io.BytesIO()
        image.convert("RGB").save(image_bytes, format="PNG")
        command = [
            str(self.executable),
            "stdin",
            "stdout",
            "--tessdata-dir",
            str(self.tessdata),
            "-l",
            self.settings.tesseract_languages,
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
            "tsv",
        ]
        completed = subprocess.run(
            command,
            input=image_bytes.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=self.settings.tesseract_timeout_seconds,
            cwd=self.executable.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Tesseract exited with code {completed.returncode}: {stderr[:300]}"
            )
        return completed.stdout.decode("utf-8", errors="replace")

    def _run_tsv(self, page: PageImage) -> str:
        return self._run_image_tsv(
            page.image,
            psm=self.settings.tesseract_page_segmentation_mode,
        )

    def _recognize_targeted_crop(
        self,
        crop: Any,
        psm: int,
    ) -> tuple[TesseractLine | None, dict[str, Any]]:
        line_started = time.perf_counter()
        try:
            parsed = parse_tesseract_tsv(self._run_image_tsv(crop.image, psm=psm))
            if not parsed:
                return None, {
                    "crop_id": crop.crop_id,
                    "psm": psm,
                    "status": "empty",
                    "elapsed_ms": round(
                        (time.perf_counter() - line_started) * 1000.0,
                        3,
                    ),
                }
            text = " ".join(line.text for line in parsed).strip()
            weights = [max(len(line.text), 1) for line in parsed]
            confidence = sum(
                line.confidence * weight
                for line, weight in zip(parsed, weights, strict=True)
            ) / sum(weights)
            result = TesseractLine(
                text=text,
                confidence=max(0.0, min(1.0, confidence)),
                bbox=crop.polygon.bbox,
            )
            return result, {
                "crop_id": crop.crop_id,
                "psm": psm,
                "status": "complete",
                "text": result.text,
                "confidence": result.confidence,
                "elapsed_ms": round(
                    (time.perf_counter() - line_started) * 1000.0,
                    3,
                ),
            }
        except Exception as exc:
            return None, {
                "crop_id": crop.crop_id,
                "psm": psm,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
                "elapsed_ms": round(
                    (time.perf_counter() - line_started) * 1000.0,
                    3,
                ),
            }

    def recognize_targeted(
        self,
        crops: Sequence[Any],
    ) -> tuple[list[TesseractLine | None], dict[str, Any]]:
        """Nhận dạng độc lập từng crop như một dòng duy nhất để cứu hộ PARTIAL."""
        started = time.perf_counter()
        psm = self.settings.partial_remediation_tesseract_psm
        metrics: dict[str, Any] = {
            "enabled": self.settings.partial_remediation_enabled,
            "psm": psm,
            "attempted_count": 0,
            "matched_count": 0,
            "failed_count": 0,
            "events": [],
        }
        if not self.settings.partial_remediation_enabled:
            metrics.update(status="disabled", elapsed_ms=0.0)
            return [None] * len(crops), metrics
        if not self._runtime_available():
            metrics.update(
                status="unavailable",
                executable=str(self.executable),
                tessdata=str(self.tessdata),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            return [None] * len(crops), metrics

        if len(crops) <= 1:
            outcomes = [self._recognize_targeted_crop(crop, psm) for crop in crops]
        else:
            with ThreadPoolExecutor(
                max_workers=min(4, len(crops)),
                thread_name_prefix="tesseract-line",
            ) as executor:
                futures = [
                    executor.submit(self._recognize_targeted_crop, crop, psm)
                    for crop in crops
                ]
                outcomes = [future.result() for future in futures]
        results = [result for result, _ in outcomes]
        metrics["attempted_count"] = len(outcomes)
        metrics["matched_count"] = sum(result is not None for result in results)
        metrics["failed_count"] = len(results) - metrics["matched_count"]
        metrics["events"] = [event for _, event in outcomes]
        metrics.update(
            status=("complete" if metrics["failed_count"] == 0 else "partial"),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return results, metrics

    def _fail_closed(
        self,
        recognitions: Sequence[Recognition],
        reason: str,
    ) -> list[Recognition]:
        if not self.settings.tesseract_fail_closed:
            return list(recognitions)
        return [_mark_high_risk(recognition, reason) for recognition in recognitions]

    def collect_page(
        self,
        page: PageImage,
    ) -> tuple[list[TesseractLine] | None, dict[str, Any]]:
        """Run independent page OCR so callers may overlap it with VietOCR."""
        started = time.perf_counter()
        metrics: dict[str, Any] = {
            "enabled": self.settings.tesseract_verifier_enabled,
            "required": self.settings.tesseract_fail_closed,
            "language": self.settings.tesseract_languages,
            "psm": self.settings.tesseract_page_segmentation_mode,
        }
        if not self.settings.tesseract_verifier_enabled:
            metrics.update(status="disabled", elapsed_ms=0.0)
            return None, metrics
        if not self._runtime_available():
            metrics.update(
                status="unavailable",
                executable=str(self.executable),
                tessdata=str(self.tessdata),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            return None, metrics

        try:
            lines = parse_tesseract_tsv(self._run_tsv(page))
        except Exception as exc:
            metrics.update(
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )
            return None, metrics
        metrics.update(
            status="complete",
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        return lines, metrics

    def verify_collected(
        self,
        crops: Sequence[Any],
        recognitions: Sequence[Recognition],
        lines: list[TesseractLine] | None,
        collection_metrics: dict[str, Any],
    ) -> tuple[list[Recognition], dict[str, Any]]:
        apply_started = time.perf_counter()
        metrics = dict(collection_metrics)
        metrics["input_line_count"] = len(recognitions)
        status = metrics.get("status")
        if status == "disabled":
            return list(recognitions), metrics
        if status != "complete" or lines is None:
            reason = (
                "tesseract_verifier_unavailable"
                if status == "unavailable"
                else "tesseract_verifier_failed"
            )
            return self._fail_closed(recognitions, reason), metrics

        crop_boxes = [crop.polygon.bbox for crop in crops[: len(recognitions)]]
        matches = match_tesseract_lines(
            crop_boxes,
            lines,
            min_score=self.settings.tesseract_line_match_score,
        )
        if len(matches) < len(recognitions):
            matches.extend([None] * (len(recognitions) - len(matches)))

        verified: list[Recognition] = []
        matched_count = 0
        low_confidence_count = 0
        disagreement_count = 0
        unmatched_count = 0
        for recognition, match in zip(recognitions, matches):
            if match is None:
                unmatched_count += 1
                verified.append(
                    _mark_high_risk(recognition, "tesseract_line_unmatched")
                    if self.settings.tesseract_fail_closed
                    else recognition
                )
                continue
            matched_count += 1
            if match.confidence < self.settings.tesseract_min_confidence:
                low_confidence_count += 1
                candidate = replace(
                    recognition,
                    verifier_text=match.text,
                    verifier_confidence=match.confidence,
                )
                verified.append(
                    _mark_high_risk(candidate, "tesseract_low_confidence")
                    if self.settings.tesseract_fail_closed
                    else candidate
                )
                continue
            candidate = apply_tesseract_verification(
                recognition,
                match,
                self.settings,
            )
            if candidate.semantic_reasons != recognition.semantic_reasons:
                disagreement_count += 1
            verified.append(candidate)

        metrics.update(
            status="complete",
            output_line_count=len(lines),
            matched_line_count=matched_count,
            unmatched_line_count=unmatched_count,
            low_confidence_line_count=low_confidence_count,
            disagreement_line_count=disagreement_count,
            elapsed_ms=round(
                float(metrics.get("elapsed_ms", 0.0))
                + (time.perf_counter() - apply_started) * 1000.0,
                3,
            ),
        )
        return verified, metrics

    def verify(
        self,
        page: PageImage,
        crops: Sequence[Any],
        recognitions: Sequence[Recognition],
    ) -> tuple[list[Recognition], dict[str, Any]]:
        lines, metrics = self.collect_page(page)
        return self.verify_collected(crops, recognitions, lines, metrics)
