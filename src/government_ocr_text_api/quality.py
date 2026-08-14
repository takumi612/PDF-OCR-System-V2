from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .config import Settings
from .models import PageImage


@dataclass(frozen=True)
class PageQualityResult:
    action: str
    estimated_skew: float
    dark_pixel_ratio: float
    low_ink: bool
    needs_review: bool = False


def _dark_pixel_ratio(image: Image.Image) -> float:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    height, width = gray.shape
    border_y = int(height * 0.03)
    border_x = int(width * 0.03)
    interior = gray[
        border_y : height - border_y if border_y else height,
        border_x : width - border_x if border_x else width,
    ]
    return float((interior < 190).mean()) if interior.size else 0.0


def _estimate_skew(image: Image.Image) -> float:
    gray = ImageOps.grayscale(image)
    if gray.width > 600:
        new_height = max(1, round(gray.height * 600 / gray.width))
        gray = gray.resize((600, new_height))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-3.0, 3.01, 0.25):
        rotated = gray.rotate(float(angle), resample=Image.Resampling.BILINEAR, fillcolor=255)
        ink = np.asarray(rotated, dtype=np.uint8) < 190
        score = float(np.var(ink.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return -best_angle


class PageQualityGate:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, page: PageImage) -> PageQualityResult:
        ratio = _dark_pixel_ratio(page.image)
        low_ink = ratio < self.settings.low_ink_ratio
        rotation = page.rotation % 360
        if rotation:
            return PageQualityResult("rotate_metadata", 0.0, ratio, low_ink)
        if page.pixel_width > page.pixel_height:
            return PageQualityResult("orientation_review", 0.0, ratio, low_ink, True)
        skew = _estimate_skew(page.image)
        if abs(skew) <= self.settings.max_skew_without_correction:
            return PageQualityResult("keep_original", skew, ratio, low_ink)
        return PageQualityResult("deskew_required", skew, ratio, low_ink)

    def apply(self, page: PageImage, quality: PageQualityResult) -> PageImage:
        image = page.image
        if quality.action == "rotate_metadata":
            image = image.rotate(-page.rotation, expand=True, fillcolor=255)
        elif quality.action == "deskew_required":
            image = image.rotate(
                -quality.estimated_skew,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=255,
            )
        if image is page.image:
            return page
        return PageImage(page_index=page.page_index, image=image, rotation=0)
