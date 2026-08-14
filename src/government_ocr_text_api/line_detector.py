from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from .config import Settings
from .models import LinePolygon, PageImage


@dataclass(frozen=True)
class _ColumnSplit:
    x: float
    gap: float
    left_count: int
    right_count: int
    y_overlap_ratio: float
    crossing_count: int
    occupancy: float
    table_like_row_ratio: float
    mode: str


def _as_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        nested = result.get("res")
        return nested if isinstance(nested, dict) else result
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if isinstance(value, dict):
        nested = value.get("res")
        return nested if isinstance(nested, dict) else value
    value = getattr(result, "res", None)
    return value if isinstance(value, dict) else {}


def _iter_payloads(results: Any) -> Iterable[dict[str, Any]]:
    if isinstance(results, dict):
        yield _as_payload(results)
        return
    for result in results:
        yield _as_payload(result)


def _row_order(polygons: list[LinePolygon], tolerance: float) -> list[LinePolygon]:
    """Sắp xếp top-to-bottom, left-to-right trong một vùng một cột."""
    if not polygons:
        return []
    rows: list[list[LinePolygon]] = []
    for polygon in sorted(polygons, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        center_y = (polygon.bbox.y0 + polygon.bbox.y1) / 2
        if not rows:
            rows.append([polygon])
            continue
        previous_center = float(
            np.mean([(item.bbox.y0 + item.bbox.y1) / 2 for item in rows[-1]])
        )
        if abs(center_y - previous_center) <= tolerance:
            rows[-1].append(polygon)
        else:
            rows.append([polygon])
    return [item for row in rows for item in sorted(row, key=lambda value: value.bbox.x0)]


def _vertical_overlap_ratio(left: LinePolygon, right: LinePolygon) -> float:
    overlap = max(0.0, min(left.bbox.y1, right.bbox.y1) - max(left.bbox.y0, right.bbox.y0))
    return overlap / max(1.0, min(left.bbox.height, right.bbox.height))


def _merge_pair(left: LinePolygon, right: LinePolygon) -> LinePolygon:
    x0 = min(left.bbox.x0, right.bbox.x0)
    y0 = min(left.bbox.y0, right.bbox.y0)
    x1 = max(left.bbox.x1, right.bbox.x1)
    y1 = max(left.bbox.y1, right.bbox.y1)
    return LinePolygon(
        points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        confidence=min(left.confidence, right.confidence),
        source="baseline_fragment_merge",
        model_version=f"{left.model_version}+{right.model_version}",
    )


def _merge_baseline_fragments(
    polygons: list[LinePolygon],
    median_height: float,
    max_gap_ratio: float,
    narrow_width_ratio: float,
) -> tuple[list[LinePolygon], int]:
    """Gộp marker/La Mã rất hẹp với dòng cùng baseline.

    Paddle đôi khi tách ``I``/``III`` trong ``Chương I`` hoặc marker ``a)`` thành
    polygon riêng. Chỉ gộp khi hai bbox chồng y mạnh, khoảng cách nhỏ và ít nhất
    một phía đủ hẹp; vì vậy các cột độc lập cùng hàng không bị nối nhầm.
    """
    if len(polygons) < 2:
        return polygons, 0
    tolerance = max(1.0, median_height * 0.22)
    rows: list[list[LinePolygon]] = []
    for polygon in sorted(polygons, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        center_y = (polygon.bbox.y0 + polygon.bbox.y1) / 2
        if not rows:
            rows.append([polygon])
            continue
        row_center = float(
            np.mean([(item.bbox.y0 + item.bbox.y1) / 2 for item in rows[-1]])
        )
        if abs(center_y - row_center) <= tolerance:
            rows[-1].append(polygon)
        else:
            rows.append([polygon])

    merged: list[LinePolygon] = []
    merge_count = 0
    max_gap = max(4.0, median_height * max_gap_ratio)
    narrow_limit = max(8.0, median_height * narrow_width_ratio)
    for row in rows:
        ordered = sorted(row, key=lambda item: item.bbox.x0)
        current = ordered[0]
        for candidate in ordered[1:]:
            gap = candidate.bbox.x0 - current.bbox.x1
            narrow = min(current.bbox.width, candidate.bbox.width) <= narrow_limit
            aligned = _vertical_overlap_ratio(current, candidate) >= 0.62
            similar_height = (
                max(current.bbox.height, candidate.bbox.height)
                / max(1.0, min(current.bbox.height, candidate.bbox.height))
                <= 1.8
            )
            if -median_height * 0.15 <= gap <= max_gap and narrow and aligned and similar_height:
                current = _merge_pair(current, candidate)
                merge_count += 1
            else:
                merged.append(current)
                current = candidate
        merged.append(current)
    return merged, merge_count


def _vertical_blocks(
    polygons: list[LinePolygon],
    median_height: float,
    page_height: float,
) -> list[list[LinePolygon]]:
    """Chia trang thành dải y để tìm layout cục bộ."""
    if not polygons:
        return []
    threshold = max(median_height * 1.25, page_height * 0.01, 8.0)
    blocks: list[list[LinePolygon]] = []
    current: list[LinePolygon] = []
    current_bottom = -1.0
    for polygon in sorted(polygons, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        if current and polygon.bbox.y0 - current_bottom > threshold:
            blocks.append(current)
            current = []
            current_bottom = -1.0
        current.append(polygon)
        current_bottom = max(current_bottom, polygon.bbox.y1)
    if current:
        blocks.append(current)
    return blocks


def _span_y(polygons: list[LinePolygon]) -> tuple[float, float]:
    return min(item.bbox.y0 for item in polygons), max(item.bbox.y1 for item in polygons)


def _gutter_eligible(
    polygon: LinePolygon,
    page_width: float,
    median_height: float,
) -> bool:
    # Con dấu/chữ ký hoặc bbox full-width không được phép xóa mất gutter chỉ vì
    # một polygon bất thường đi ngang qua vùng giữa hai cột.
    if polygon.bbox.height > median_height * 2.6:
        return False
    if polygon.bbox.width > page_width * 0.80:
        return False
    if polygon.bbox.area > page_width * median_height * 4.5:
        return False
    return True


def _low_occupancy_runs(
    polygons: list[LinePolygon],
    page_width: float,
    median_height: float,
    occupancy_threshold: float,
    min_gap_ratio: float,
) -> list[tuple[float, float, float]]:
    eligible = [
        polygon
        for polygon in polygons
        if _gutter_eligible(polygon, page_width, median_height)
    ]
    if len(eligible) < 4:
        return []

    bin_count = max(80, min(256, int(round(page_width / 7.0))))
    occupancy = np.zeros(bin_count, dtype=np.float32)
    for polygon in eligible:
        left = max(0, min(bin_count - 1, int(polygon.bbox.x0 / page_width * bin_count)))
        right = max(left + 1, min(bin_count, int(np.ceil(polygon.bbox.x1 / page_width * bin_count))))
        occupancy[left:right] += 1.0
    occupancy /= max(1, len(eligible))

    # Chỉ xét vùng giữa trang; lề trái/phải không phải gutter cột.
    low = occupancy <= occupancy_threshold
    central_left = int(bin_count * 0.22)
    central_right = int(bin_count * 0.84)
    low[:central_left] = False
    low[central_right:] = False

    minimum_gap = max(page_width * min_gap_ratio, median_height * 1.2, 10.0)
    minimum_bins = max(2, int(np.ceil(minimum_gap / page_width * bin_count)))
    runs: list[tuple[float, float, float]] = []
    start: int | None = None
    for index, value in enumerate(low.tolist() + [False]):
        if value and start is None:
            start = index
            continue
        if value or start is None:
            continue
        end = index
        if end - start >= minimum_bins:
            x0 = start / bin_count * page_width
            x1 = end / bin_count * page_width
            mean_occupancy = float(np.mean(occupancy[start:end]))
            runs.append((x0, x1, mean_occupancy))
        start = None
    return runs




def _table_like_row_ratio(polygons: list[LinePolygon], median_height: float) -> float:
    if not polygons:
        return 0.0
    tolerance = max(1.0, median_height * 0.22)
    rows: list[list[LinePolygon]] = []
    for polygon in sorted(polygons, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        center = (polygon.bbox.y0 + polygon.bbox.y1) / 2
        if not rows:
            rows.append([polygon])
            continue
        row_center = float(np.mean([(item.bbox.y0 + item.bbox.y1) / 2 for item in rows[-1]]))
        if abs(center - row_center) <= tolerance:
            rows[-1].append(polygon)
        else:
            rows.append([polygon])
    # Hai ô cùng baseline lặp lại nhiều lần là dấu hiệu bảng/biểu mẫu.
    # Ngưỡng cũ >=3 bỏ lọt các form hai cột và gây reorder sai toàn trang.
    return sum(len(row) >= 2 for row in rows) / max(1, len(rows))

def _detect_column_split(
    polygons: list[LinePolygon],
    page_width: float,
    median_height: float,
    page_height: float | None = None,
    *,
    occupancy_threshold: float = 0.14,
    min_gap_ratio: float = 0.03,
    signature_enabled: bool = True,
    signature_max_right_lines: int = 6,
) -> _ColumnSplit | None:
    """Tìm gutter bằng mật độ phủ thay vì union tuyệt đối.

    Một dòng danh sách dài hoặc polygon con dấu có thể đi qua gutter. Mô hình
    occupancy chấp nhận một lượng phủ nhỏ và vẫn phát hiện khối chữ ký bất cân
    bằng (nhiều dòng bên trái, 2--5 dòng bên phải).
    """
    if len(polygons) < 4:
        return None
    runs = _low_occupancy_runs(
        polygons,
        page_width,
        median_height,
        occupancy_threshold,
        min_gap_ratio,
    )
    if not runs:
        return None

    best: tuple[float, _ColumnSplit] | None = None
    table_like_ratio = _table_like_row_ratio(polygons, median_height)
    for left_edge, right_edge, occupancy in runs:
        split_x = (left_edge + right_edge) / 2
        gap = right_edge - left_edge
        left = [item for item in polygons if (item.bbox.x0 + item.bbox.x1) / 2 < split_x]
        right = [item for item in polygons if (item.bbox.x0 + item.bbox.x1) / 2 >= split_x]
        crossing = [item for item in polygons if item.bbox.x0 < split_x < item.bbox.x1]
        if len(left) < 2 or len(right) < 2:
            continue

        left_y0, left_y1 = _span_y(left)
        right_y0, right_y1 = _span_y(right)
        overlap = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
        overlap_ratio = overlap / max(1.0, min(left_y1 - left_y0, right_y1 - right_y0))
        count_balance = min(len(left), len(right)) / max(len(left), len(right))
        crossing_ratio = len(crossing) / max(1, len(polygons))

        normal_columns = (
            count_balance >= 0.16
            and overlap_ratio >= 0.24
            and crossing_ratio <= 0.24
            and table_like_ratio < 0.30
        )
        right_center = float(
            np.mean([(item.bbox.x0 + item.bbox.x1) / 2 for item in right])
        )
        signature_block = (
            signature_enabled
            and len(left) >= 5
            and 2 <= len(right) <= signature_max_right_lines
            and right_center >= page_width * 0.58
            and overlap_ratio >= 0.12
            and crossing_ratio <= 0.35
            and table_like_ratio < 0.30
        )
        if not normal_columns and not signature_block:
            continue

        mode = (
            "signature_block"
            if signature_block and len(left) >= len(right) * 1.8
            else "columns"
        )
        split = _ColumnSplit(
            x=split_x,
            gap=gap,
            left_count=len(left),
            right_count=len(right),
            y_overlap_ratio=overlap_ratio,
            crossing_count=len(crossing),
            occupancy=occupancy,
            table_like_row_ratio=table_like_ratio,
            mode=mode,
        )
        score = gap * (1.0 - occupancy) * (0.55 + overlap_ratio)
        if mode == "signature_block":
            score *= 1.2
        else:
            score *= 0.75 + count_balance
        if best is None or score > best[0]:
            best = (score, split)
    return best[1] if best else None


def _signature_crossing_affiliates_right(
    item: LinePolygon,
    split_x: float,
    right_anchors: list[LinePolygon],
    page_width: float,
    median_height: float,
    boundary_y: float | None = None,
) -> bool:
    """Gán dòng cắt gutter vào cụm chữ ký phải khi hình học đủ rõ.

    Dòng ``KT. THỦ TƯỚNG`` trên tài liệu thật thường bị Paddle kéo bbox sang
    trái do con dấu, nên tâm bbox rơi vào cột Nơi nhận. Quy tắc này chỉ áp
    dụng trong ``signature_block`` và yêu cầu dòng compact, bắt đầu gần cụm
    anchor phải, có phần đáng kể nằm bên phải gutter và gần anchor theo trục y.
    Các dòng Nơi nhận dài bắt đầu từ lề trái hoặc bbox con dấu lớn không đạt
    các điều kiện này.
    """
    if not right_anchors or not (item.bbox.x0 < split_x < item.bbox.x1):
        return False
    if (
        boundary_y is not None
        and item.bbox.y1 < boundary_y - median_height * 0.8
    ):
        return False
    if item.bbox.height > median_height * 2.2:
        return False
    if item.bbox.width > page_width * 0.50:
        return False

    right_overlap = max(0.0, item.bbox.x1 - split_x)
    right_fraction = right_overlap / max(1.0, item.bbox.width)
    if right_fraction < 0.18:
        return False

    anchor_left = min(anchor.bbox.x0 for anchor in right_anchors)
    anchor_top = min(anchor.bbox.y0 for anchor in right_anchors)
    anchor_bottom = max(anchor.bbox.y1 for anchor in right_anchors)
    allowed_left_shift = max(page_width * 0.28, median_height * 5.0)
    if item.bbox.x0 < max(page_width * 0.38, anchor_left - allowed_left_shift):
        return False

    y_margin = median_height * 1.8
    return (
        item.bbox.y1 >= anchor_top - y_margin
        and item.bbox.y0 <= anchor_bottom + y_margin
    )


def _order_block(
    polygons: list[LinePolygon],
    split: _ColumnSplit | None,
    tolerance: float,
    page_width: float,
    median_height: float,
) -> tuple[list[LinePolygon], dict[str, float | int | None]]:
    if split is None:
        return _row_order(polygons, tolerance), {
            "signature_boundary_y": None,
            "signature_prefix_count": 0,
            "signature_crossing_to_right_count": 0,
            "signature_decorative_polygon_count": 0,
        }

    left: list[LinePolygon] = []
    right: list[LinePolygon] = []
    prefix: list[LinePolygon] = []
    suffix: list[LinePolygon] = []
    decorations: list[LinePolygon] = []

    stable_left = [item for item in polygons if item.bbox.x1 <= split.x]
    stable_right = [item for item in polygons if item.bbox.x0 >= split.x]
    compact_right_anchors = [
        item
        for item in stable_right
        if item.bbox.height <= median_height * 2.2
        and item.bbox.width <= page_width * 0.55
    ]

    # Với khối chữ ký bất cân bằng, cột phải là anchor đáng tin cậy nhất để
    # xác định nơi layout hai cột thực sự bắt đầu. Bản 1.3.1 dùng min-y của cả
    # stable_left và stable_right, khiến các dòng thân bài ở cột trái kéo
    # ``column_top`` lên quá cao; sau đó toàn bộ thân bài bị tách trái/phải và
    # đổi thứ tự. Chỉ lấy top của cột phải giữ nguyên row-order cho prefix,
    # đồng thời vẫn cho phép đọc hết danh sách Nơi nhận trước khối chữ ký.
    signature_boundary_y: float | None = None
    if split.mode == "signature_block":
        right_anchors = compact_right_anchors or [
            item
            for item in polygons
            if (item.bbox.x0 + item.bbox.x1) / 2 >= split.x
            and item.bbox.width <= page_width * 0.55
            and item.bbox.height <= median_height * 2.2
        ]
        if right_anchors:
            signature_boundary_y = min(item.bbox.y0 for item in right_anchors)

    if signature_boundary_y is not None:
        # Chỉ những dòng kết thúc rõ ràng trước anchor phải mới thuộc prefix.
        # Khoảng đệm nhỏ tránh đẩy dòng cùng baseline với ``KT./BỘ TRƯỞNG``
        # vào prefix và làm xen thứ tự hai cột.
        boundary_margin = max(1.0, tolerance * 0.20)
        prefix_ids = {
            id(item)
            for item in polygons
            if item.bbox.y1 <= signature_boundary_y - boundary_margin
            and not _signature_crossing_affiliates_right(
                item,
                split.x,
                compact_right_anchors,
                page_width,
                median_height,
                signature_boundary_y,
            )
        }
    else:
        prefix_ids = set()

    remaining = [item for item in polygons if id(item) not in prefix_ids]
    prefix = [item for item in polygons if id(item) in prefix_ids]

    remaining_stable = [
        item
        for item in remaining
        if item.bbox.x1 <= split.x or item.bbox.x0 >= split.x
    ]
    column_top = min(
        [item.bbox.y0 for item in remaining_stable],
        default=min(item.bbox.y0 for item in remaining),
    )
    column_bottom = max(
        [item.bbox.y1 for item in remaining_stable],
        default=max(item.bbox.y1 for item in remaining),
    )

    signature_crossing_to_right_count = 0
    for item in remaining:
        crosses = item.bbox.x0 < split.x < item.bbox.x1
        crossing_to_right = (
            split.mode == "signature_block"
            and _signature_crossing_affiliates_right(
                item,
                split.x,
                compact_right_anchors,
                page_width,
                median_height,
                signature_boundary_y,
            )
        )
        is_full_width_text = (
            item.bbox.width > page_width * 0.66
            and item.bbox.height <= median_height * 2.2
        )
        is_signature_decoration = (
            split.mode == "signature_block"
            and crosses
            and (
                item.bbox.height > median_height * 2.2
                or item.bbox.area > page_width * median_height * 4.5
            )
        )
        if is_signature_decoration:
            decorations.append(item)
            continue
        if (
            signature_boundary_y is None
            and crosses
            and is_full_width_text
            and item.bbox.y1 <= column_top
        ):
            prefix.append(item)
            continue
        if crosses and is_full_width_text and item.bbox.y0 >= column_bottom:
            suffix.append(item)
            continue
        if crossing_to_right:
            right.append(item)
            signature_crossing_to_right_count += 1
            continue
        center_x = (item.bbox.x0 + item.bbox.x1) / 2
        if center_x < split.x:
            left.append(item)
        else:
            right.append(item)

    return (
        _row_order(prefix, tolerance)
        + _row_order(left, tolerance)
        + _row_order(right, tolerance)
        + _row_order(decorations, tolerance)
        + _row_order(suffix, tolerance),
        {
            "signature_boundary_y": signature_boundary_y,
            "signature_prefix_count": len(prefix_ids),
            "signature_crossing_to_right_count": signature_crossing_to_right_count,
            "signature_decorative_polygon_count": len(decorations),
        },
    )


def _column_aware_order(
    polygons: list[LinePolygon],
    page_width: float,
    page_height: float,
    settings: Settings | None = None,
) -> tuple[list[LinePolygon], dict[str, Any]]:
    if not polygons:
        return [], {
            "layout_mode": "empty",
            "vertical_block_count": 0,
            "column_block_count": 0,
            "column_gutters": [],
            "column_modes": [],
            "signature_boundary_y": [],
            "signature_prefix_count": [],
            "signature_crossing_to_right_count": [],
            "signature_decorative_polygon_count": [],
        }

    settings = settings or Settings()
    median_height = float(np.median([polygon.bbox.height for polygon in polygons]))
    tolerance = max(median_height / 2.0, 1.0)
    blocks = _vertical_blocks(polygons, median_height, page_height)
    ordered: list[LinePolygon] = []
    splits: list[_ColumnSplit] = []
    ordering_details: list[dict[str, float | int | None]] = []
    for block in blocks:
        split = _detect_column_split(
            block,
            page_width,
            median_height,
            page_height,
            occupancy_threshold=settings.ocr_column_occupancy_threshold,
            min_gap_ratio=settings.ocr_column_min_gap_ratio,
            signature_enabled=settings.ocr_signature_block_enabled,
            signature_max_right_lines=settings.ocr_signature_block_max_right_lines,
        )
        if split is not None:
            splits.append(split)
        block_ordered, block_details = _order_block(
            block,
            split,
            tolerance,
            page_width,
            median_height,
        )
        ordered.extend(block_ordered)
        if split is not None:
            ordering_details.append(block_details)

    if any(split.mode == "signature_block" for split in splits):
        layout_mode = "signature_block"
    elif splits:
        layout_mode = "column_aware"
    else:
        layout_mode = "row"
    return ordered, {
        "layout_mode": layout_mode,
        "vertical_block_count": len(blocks),
        "column_block_count": len(splits),
        "column_gutters": [round(split.x, 2) for split in splits],
        "column_gap_pixels": [round(split.gap, 2) for split in splits],
        "column_left_counts": [split.left_count for split in splits],
        "column_right_counts": [split.right_count for split in splits],
        "column_y_overlap_ratio": [round(split.y_overlap_ratio, 3) for split in splits],
        "column_crossing_count": [split.crossing_count for split in splits],
        "column_occupancy": [round(split.occupancy, 4) for split in splits],
        "column_table_like_row_ratio": [
            round(split.table_like_row_ratio, 3) for split in splits
        ],
        "column_modes": [split.mode for split in splits],
        "signature_boundary_y": [
            (
                round(float(detail["signature_boundary_y"]), 2)
                if detail["signature_boundary_y"] is not None
                else None
            )
            for detail in ordering_details
        ],
        "signature_prefix_count": [
            int(detail["signature_prefix_count"]) for detail in ordering_details
        ],
        "signature_crossing_to_right_count": [
            int(detail["signature_crossing_to_right_count"])
            for detail in ordering_details
        ],
        "signature_decorative_polygon_count": [
            int(detail["signature_decorative_polygon_count"])
            for detail in ordering_details
        ],
    }


def _preliminary_order(polygons: list[LinePolygon]) -> list[LinePolygon]:
    """Backward-compatible row ordering used by older tests/callers."""
    if not polygons:
        return []
    median_height = float(np.median([polygon.bbox.height for polygon in polygons]))
    return _row_order(polygons, max(median_height / 2.0, 1.0))


class PaddleLineDetector:
    """PaddleOCR chỉ phát hiện đa giác dòng; VietOCR là recognizer duy nhất."""

    def __init__(self, settings: Settings, processor: Any | None = None) -> None:
        self.settings = settings
        self.processor = processor
        self.last_metrics: dict[str, Any] = {}
        self.init_metrics: dict[str, Any] = {
            "provided_processor": processor is not None,
            "mkldnn_requested": bool(settings.paddle_enable_mkldnn),
            "mkldnn_effective": None,
            "cpu_threads_requested": settings.paddle_cpu_threads,
            "cpu_threads_effective": None,
            "fallback_used": False,
            "fallback_error_type": None,
            "fallback_phase": None,
        }

    def warm(self) -> None:
        self._processor()

    @staticmethod
    def _construct_text_detection(TextDetection: Any, kwargs: dict[str, Any]) -> tuple[Any, bool]:
        """Construct Paddle TextDetection while tolerating older wrappers.

        PaddleOCR 3.x exposes ``cpu_threads`` but downstream wrappers may not.
        Retry only the unsupported keyword case; model/runtime failures are left
        to the caller so MKL-DNN fallback can be handled separately.
        """
        try:
            return TextDetection(**kwargs), "cpu_threads" in kwargs
        except TypeError as exc:
            if "cpu_threads" not in kwargs or "cpu_threads" not in str(exc):
                raise
            reduced = dict(kwargs)
            reduced.pop("cpu_threads", None)
            return TextDetection(**reduced), False

    @staticmethod
    def _is_onednn_pir_compatibility_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            isinstance(exc, NotImplementedError)
            and "convertpirattribute2runtimeattribute" in message
            and "onednn" in message
        )

    def _create_processor(self, enable_mkldnn: bool) -> Any:
        # Trên Windows, nạp Torch trước Paddle để tránh xung đột DLL.
        import torch  # noqa: F401
        from paddleocr import TextDetection

        kwargs: dict[str, Any] = {
            "model_name": self.settings.line_detection_model_name,
            "device": self.settings.paddle_device,
        }
        if not self.settings.paddle_device.startswith("gpu"):
            kwargs["enable_mkldnn"] = enable_mkldnn
            kwargs["cpu_threads"] = int(self.settings.paddle_cpu_threads)

        processor, cpu_threads_effective = self._construct_text_detection(
            TextDetection,
            kwargs,
        )
        self.init_metrics["mkldnn_effective"] = kwargs.get("enable_mkldnn")
        self.init_metrics["cpu_threads_effective"] = (
            self.settings.paddle_cpu_threads if cpu_threads_effective else None
        )
        return processor

    def _processor(self) -> Any:
        if self.processor is None:
            enable_mkldnn = bool(
                self.settings.paddle_enable_mkldnn
                and not self.settings.paddle_device.startswith("gpu")
            )
            try:
                self.processor = self._create_processor(enable_mkldnn)
            except Exception as exc:
                can_fallback = bool(
                    self.settings.paddle_mkldnn_fallback
                    and enable_mkldnn
                )
                if not can_fallback:
                    raise
                self.processor = self._create_processor(False)
                self.init_metrics["fallback_used"] = True
                self.init_metrics["fallback_error_type"] = type(exc).__name__
                self.init_metrics["fallback_phase"] = "init"
        return self.processor

    def detect(self, page: PageImage) -> list[LinePolygon]:
        processor = self._processor()
        image = np.asarray(page.image)
        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        try:
            results = processor.predict(image)
        except Exception as exc:
            can_fallback = bool(
                not self.init_metrics["provided_processor"]
                and self.settings.paddle_mkldnn_fallback
                and self.init_metrics["mkldnn_effective"] is True
                and self._is_onednn_pir_compatibility_error(exc)
            )
            if not can_fallback:
                raise
            processor = self._create_processor(False)
            self.processor = processor
            self.init_metrics["fallback_used"] = True
            self.init_metrics["fallback_error_type"] = type(exc).__name__
            self.init_metrics["fallback_phase"] = "predict"
            results = processor.predict(image)
        model_version = getattr(
            processor, "model_version", self.settings.line_detection_model_name
        )
        candidates: list[LinePolygon] = []
        for payload in _iter_payloads(results):
            raw_polygons = payload.get("dt_polys")
            if raw_polygons is None:
                raw_polygons = payload.get("polys")
            if raw_polygons is None:
                raw_polygons = []
            scores = payload.get("dt_scores")
            if scores is None:
                scores = payload.get("scores")
            if scores is None:
                scores = []
            for index, raw_polygon in enumerate(raw_polygons):
                points = np.asarray(raw_polygon, dtype=np.float32)
                if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 2:
                    continue
                x0, y0 = points.min(axis=0)
                x1, y1 = points.max(axis=0)
                width = float(x1 - x0)
                height = float(y1 - y0)
                area = float(cv2.contourArea(points))
                normal_line = width >= 8 and height >= 6 and area >= 96
                # Ký tự La Mã ``I`` hoặc dấu marker rất hẹp có thể chỉ rộng
                # 3--7 px ở 200 DPI. Giữ tạm các bbox cao-hẹp để thử ghép vào
                # polygon cùng baseline; fragment không ghép được sẽ bị loại sau.
                narrow_fragment = (
                    width >= 3
                    and height >= 10
                    and area >= 28
                    and height / max(width, 1.0) >= 1.35
                )
                if not normal_line and not narrow_fragment:
                    continue
                confidence = float(scores[index]) if index < len(scores) else 1.0
                candidates.append(
                    LinePolygon(
                        points=[(float(x), float(y)) for x, y in points],
                        confidence=confidence,
                        source=(
                            "paddle_line_detector"
                            if normal_line
                            else "paddle_line_detector_narrow_fragment"
                        ),
                        model_version=str(model_version),
                    )
                )
        selected: list[LinePolygon] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if any(candidate.bbox.iou(existing.bbox) > 0.85 for existing in selected):
                continue
            selected.append(candidate)

        fragment_merge_count = 0
        narrow_fragment_dropped_count = 0
        if selected and self.settings.merge_baseline_fragments:
            regular_heights = [
                polygon.bbox.height
                for polygon in selected
                if polygon.source != "paddle_line_detector_narrow_fragment"
            ]
            median_height = float(
                np.median(regular_heights or [polygon.bbox.height for polygon in selected])
            )
            selected, fragment_merge_count = _merge_baseline_fragments(
                selected,
                median_height,
                self.settings.baseline_fragment_max_gap_ratio,
                self.settings.baseline_fragment_narrow_width_ratio,
            )
            before_drop = len(selected)
            selected = [
                polygon
                for polygon in selected
                if polygon.source != "paddle_line_detector_narrow_fragment"
            ]
            narrow_fragment_dropped_count = before_drop - len(selected)
        else:
            # Khi merge bị tắt, không để các fragment cao-hẹp trở thành dòng rác.
            before_drop = len(selected)
            selected = [
                polygon
                for polygon in selected
                if polygon.source != "paddle_line_detector_narrow_fragment"
            ]
            narrow_fragment_dropped_count = before_drop - len(selected)

        if self.settings.ocr_column_aware_ordering:
            ordered, layout_metrics = _column_aware_order(
                selected,
                float(page.pixel_width),
                float(page.pixel_height),
                self.settings,
            )
        else:
            ordered = _preliminary_order(selected)
            layout_metrics = {
                "layout_mode": "row_disabled",
                "vertical_block_count": 1 if selected else 0,
                "column_block_count": 0,
                "column_gutters": [],
                "column_modes": [],
                "signature_boundary_y": [],
                "signature_prefix_count": [],
                "signature_crossing_to_right_count": [],
                "signature_decorative_polygon_count": [],
            }
        self.last_metrics = {
            "detector_runtime": dict(self.init_metrics),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "baseline_fragment_merge_count": fragment_merge_count,
            "narrow_fragment_dropped_count": narrow_fragment_dropped_count,
            **layout_metrics,
        }
        return ordered
