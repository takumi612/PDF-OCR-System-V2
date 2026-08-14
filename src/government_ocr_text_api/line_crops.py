from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .models import LineCrop, LinePolygon, PageImage


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]      # top-left
    rect[2] = points[np.argmax(sums)]      # bottom-right
    rect[1] = points[np.argmin(diffs)]     # top-right
    rect[3] = points[np.argmax(diffs)]     # bottom-left
    return rect


def make_line_crop(
    page: PageImage,
    polygon: LinePolygon,
    padding_ratio: float,
    crop_id: str,
) -> LineCrop:
    points = np.asarray(polygon.points, dtype=np.float32)
    if points.shape[0] != 4:
        box = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float32)
    else:
        box = points
    rect = _ordered_quad(box)
    tl, tr, br, bl = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)
    pad_x = max(1, round(width * padding_ratio))
    pad_y = max(1, round(height * padding_ratio))
    target = np.array(
        [
            [pad_x, pad_y],
            [pad_x + width - 1, pad_y],
            [pad_x + width - 1, pad_y + height - 1],
            [pad_x, pad_y + height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, target)
    source = np.asarray(page.image.convert("RGB"))
    warped = cv2.warpPerspective(
        source,
        matrix,
        (width + 2 * pad_x, height + 2 * pad_y),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    image = Image.fromarray(warped).convert("RGB")
    return LineCrop(crop_id=crop_id, image=image, polygon=polygon)


def make_horizontal_expanded_line_crop(
    page: PageImage,
    crop: LineCrop,
    right_expand_height_ratio: float,
    crop_id: str,
) -> LineCrop:
    """Tạo crop retry mở rộng sang phải theo bbox gốc.

    Dùng cho tiêu đề ngắn như ``Chương`` khi detector có thể bỏ sót ký tự La Mã
    rất hẹp nằm ngay bên phải. Chỉ mở rộng theo trục x, giữ dải y của dòng để
    không kéo nội dung dòng dưới vào lần nhận dạng lại.
    """
    bbox = crop.polygon.bbox
    height = max(1.0, bbox.height)
    pad_x = max(1, round(height * 0.12))
    pad_y = max(1, round(height * 0.12))
    expand_right = max(1, round(height * right_expand_height_ratio))

    x0 = max(0, int(np.floor(bbox.x0)) - pad_x)
    y0 = max(0, int(np.floor(bbox.y0)) - pad_y)
    x1 = min(page.pixel_width, int(np.ceil(bbox.x1)) + pad_x + expand_right)
    y1 = min(page.pixel_height, int(np.ceil(bbox.y1)) + pad_y)
    image = page.image.convert("RGB").crop((x0, y0, x1, y1)).convert("RGB")
    polygon = LinePolygon(
        points=[
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x1), float(y1)),
            (float(x0), float(y1)),
        ],
        confidence=crop.polygon.confidence,
        source="chapter_heading_expanded_retry",
        model_version=crop.polygon.model_version,
    )
    return LineCrop(crop_id=crop_id, image=image, polygon=polygon)


def make_axis_aligned_retry_crop(
    page: PageImage,
    crop: LineCrop,
    padding_height_ratio: float,
) -> LineCrop:
    """Build a conservative page-axis crop for semantic recognition retries.

    The primary recognizer keeps its perspective-normalized crop. This alternate
    view is only used after independent verifiers flag a line; avoiding a second
    perspective warp preserves clipped or distorted characters near long-line
    boundaries.
    """
    bbox = crop.polygon.bbox
    height = max(1.0, bbox.height)
    padding = max(1, round(height * padding_height_ratio))
    x0 = max(0, int(np.floor(bbox.x0)) - padding)
    y0 = max(0, int(np.floor(bbox.y0)) - padding)
    x1 = min(page.pixel_width, int(np.ceil(bbox.x1)) + padding)
    y1 = min(page.pixel_height, int(np.ceil(bbox.y1)) + padding)
    image = page.image.convert("RGB").crop((x0, y0, x1, y1)).convert("RGB")
    polygon = LinePolygon(
        points=[
            (float(x0), float(y0)),
            (float(x1), float(y0)),
            (float(x1), float(y1)),
            (float(x0), float(y1)),
        ],
        confidence=crop.polygon.confidence,
        source="semantic_retry_axis_aligned",
        model_version=crop.polygon.model_version,
    )
    return LineCrop(crop_id=crop.crop_id, image=image, polygon=polygon)
