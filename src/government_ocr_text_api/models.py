from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class LinePolygon:
    points: list[tuple[float, float]]
    confidence: float
    source: str
    model_version: str

    @property
    def bbox(self) -> BBox:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return BBox(min(xs), min(ys), max(xs), max(ys))


@dataclass
class PageImage:
    page_index: int
    image: Image.Image
    rotation: int = 0

    @property
    def pixel_width(self) -> int:
        return self.image.width

    @property
    def pixel_height(self) -> int:
        return self.image.height


@dataclass
class LineCrop:
    crop_id: str
    image: Image.Image
    polygon: LinePolygon


@dataclass
class Recognition:
    text: str
    confidence: float
    error_code: str | None = None
    message_vi: str | None = None
    raw_text: str | None = None
    semantic_risk: Literal["none", "medium", "high"] = "none"
    semantic_reasons: tuple[str, ...] = ()
    secondary_confidence: float | None = None
    verifier_text: str | None = None
    verifier_confidence: float | None = None


@dataclass
class NativePage:
    page_index: int
    markdown: str
    needs_ocr: bool
    ocr_reason: str | None = None


@dataclass
class NativeDocument:
    pages: list[NativePage]
    pages_with_tables: list[int] = field(default_factory=list)
    pages_with_columns: list[int] = field(default_factory=list)
    native_processing_ms: float = 0.0
    fallback_reason: str | None = None


class OcrLineResult(BaseModel):
    line_index: int
    crop_id: str | None = None
    text: str
    raw_text: str | None = None
    confidence: float
    error_code: str | None = None
    semantic_risk: Literal["none", "medium", "high"] = "none"
    semantic_reasons: list[str] = Field(default_factory=list)
    secondary_confidence: float | None = None
    verifier_text: str | None = None
    verifier_confidence: float | None = None
    bbox: list[float] | None = None


class PageResult(BaseModel):
    page_index: int
    page_number: int
    source: Literal["native", "ocr"]
    text: str
    markdown: str | None = None
    needs_ocr: bool
    ocr_reason: str | None = None
    confidence_mean: float | None = None
    needs_review: bool = False
    line_count: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_codes: list[str] = Field(default_factory=list)
    ai_safe_text: str = ""
    ai_ready: bool = True
    semantic_risk_count: int = 0
    line_results: list[OcrLineResult] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    filename: str
    sha256: str
    page_count: int
    native_page_count: int
    ocr_page_count: int
    status: Literal["complete", "partial"]
    processing_time_ms: float
    text: str
    markdown: str | None = None
    pages: list[PageResult] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    ai_safe_text: str = ""
    ai_ready: bool = True
    semantic_risk_count: int = 0


class ErrorResponse(BaseModel):
    error_code: str
    message_vi: str
