from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from government_ocr_text_api.config import Settings
from government_ocr_text_api.line_detector import PaddleLineDetector, _column_aware_order
from government_ocr_text_api.models import LinePolygon, PageImage


class FakeProcessor:
    model_version = "test"

    def predict(self, image):
        return [
            {
                "res": {
                    "dt_polys": np.asarray(
                        [[[10, 10], [110, 10], [110, 30], [10, 30]]],
                        dtype=np.float32,
                    ),
                    "dt_scores": np.asarray([0.98], dtype=np.float32),
                }
            }
        ]


def _poly(x0: float, y0: float, x1: float, y1: float) -> LinePolygon:
    return LinePolygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        0.99,
        "test",
        "test",
    )


def test_detector_accepts_numpy_arrays_without_truth_value_error():
    detector = PaddleLineDetector(Settings(), processor=FakeProcessor())
    page = PageImage(page_index=0, image=Image.new("RGB", (200, 100), "white"), rotation=0)

    polygons = detector.detect(page)

    assert len(polygons) == 1
    assert polygons[0].confidence > 0.9


def test_column_aware_order_keeps_signature_block_after_recipient_column():
    body = _poly(100, 40, 900, 70)
    left = [
        _poly(100, 300, 420, 325),
        _poly(100, 340, 430, 365),
        _poly(100, 380, 410, 405),
        _poly(100, 420, 440, 445),
    ]
    right = [
        _poly(620, 310, 900, 335),
        _poly(650, 350, 890, 375),
        _poly(680, 390, 870, 415),
    ]

    ordered, metrics = _column_aware_order(
        [body, left[0], right[0], left[1], right[1], left[2], right[2], left[3]],
        page_width=1000,
        page_height=1200,
    )

    assert ordered[0] is body
    assert ordered[1:5] == left
    assert ordered[5:] == right
    assert metrics["layout_mode"] == "column_aware"
    assert metrics["column_block_count"] == 1


def test_single_column_order_is_unchanged():
    polygons = [
        _poly(120, 90, 850, 115),
        _poly(120, 30, 850, 55),
        _poly(120, 60, 850, 85),
    ]
    ordered, metrics = _column_aware_order(polygons, 1000, 1200)
    assert [item.bbox.y0 for item in ordered] == [30, 60, 90]
    assert metrics["layout_mode"] == "row"


def test_page9_measured_geometry_uses_signature_block_despite_stamp_and_long_line():
    # Tọa độ xấp xỉ đo trực tiếp từ ảnh render 1673x2353 của trang 9
    # QĐ 30/2025. Hai dòng thân bài nằm trên, danh sách Nơi nhận ở trái,
    # ba dòng chữ ký ở phải và một bbox con dấu lớn cắt qua gutter.
    body = [
        _poly(344, 205, 1510, 248),
        _poly(266, 243, 936, 290),
    ]
    left = [
        _poly(282, 369 + i * 36, 1010 if i in {3, 12} else 760, 399 + i * 36)
        for i in range(16)
    ]
    right = [
        _poly(1110, 369, 1450, 402),
        _poly(1090, 407, 1455, 440),
        _poly(1110, 690, 1460, 730),
    ]
    stamp_like = _poly(760, 369, 1592, 715)

    ordered, metrics = _column_aware_order(
        body + left + right + [stamp_like],
        page_width=1673,
        page_height=2353,
    )

    assert ordered[:2] == body
    assert ordered[2 : 2 + len(left)] == left
    assert {id(item) for item in ordered[2 + len(left) :]} == {
        id(item) for item in right + [stamp_like]
    }
    assert metrics["layout_mode"] == "signature_block"
    assert metrics["column_block_count"] == 1
    assert metrics["column_modes"] == ["signature_block"]
    assert metrics["column_occupancy"][0] <= 0.14
    assert metrics["column_crossing_count"][0] >= 1


def test_baseline_fragment_merge_joins_narrow_roman_numeral():
    from government_ocr_text_api.line_detector import _merge_baseline_fragments

    chapter = _poly(300, 100, 470, 130)
    roman = _poly(480, 102, 490, 128)
    other = _poly(300, 160, 700, 190)

    merged, count = _merge_baseline_fragments(
        [chapter, roman, other],
        median_height=30,
        max_gap_ratio=1.0,
        narrow_width_ratio=1.8,
    )

    assert count == 1
    assert len(merged) == 2
    assert merged[0].bbox.x0 == 300
    assert merged[0].bbox.x1 == 490


def test_table_like_grid_is_not_reordered_as_columns():
    polygons = []
    for row in range(5):
        y0 = 200 + row * 40
        polygons.extend(
            [
                _poly(100, y0, 280, y0 + 25),
                _poly(360, y0, 620, y0 + 25),
                _poly(700, y0, 930, y0 + 25),
            ]
        )

    ordered, metrics = _column_aware_order(polygons, 1000, 1200)

    assert metrics["layout_mode"] == "row"
    assert [item.bbox.y0 for item in ordered[:3]] == [200, 200, 200]


class NarrowRomanProcessor:
    model_version = "test"

    def predict(self, image):
        return [
            {
                "res": {
                    "dt_polys": np.asarray(
                        [
                            [[300, 100], [470, 100], [470, 130], [300, 130]],
                            [[480, 102], [485, 102], [485, 128], [480, 128]],
                            [[50, 500], [55, 500], [55, 526], [50, 526]],
                        ],
                        dtype=np.float32,
                    ),
                    "dt_scores": np.asarray([0.99, 0.97, 0.60], dtype=np.float32),
                }
            }
        ]


def test_detector_keeps_narrow_roman_until_merge_and_drops_isolated_noise():
    detector = PaddleLineDetector(Settings(), processor=NarrowRomanProcessor())
    page = PageImage(
        page_index=0,
        image=Image.new("RGB", (1000, 700), "white"),
        rotation=0,
    )

    polygons = detector.detect(page)

    assert len(polygons) == 1
    assert polygons[0].bbox.x0 == 300
    assert polygons[0].bbox.x1 == 485
    assert detector.last_metrics["baseline_fragment_merge_count"] == 1
    assert detector.last_metrics["narrow_fragment_dropped_count"] == 1


def test_two_column_form_rows_are_not_treated_as_signature_block():
    polygons = []
    for row in range(10):
        y0 = 180 + row * 42
        polygons.extend(
            [
                _poly(100, y0, 470, y0 + 26),
                _poly(620, y0, 930, y0 + 26),
            ]
        )

    _, metrics = _column_aware_order(polygons, 1000, 1200)

    assert metrics["layout_mode"] == "row"
    assert metrics["column_block_count"] == 0


def test_signature_metrics_expose_column_counts():
    left = [_poly(100, 300 + i * 35, 520, 325 + i * 35) for i in range(10)]
    right = [
        _poly(650, 310, 900, 335),
        _poly(670, 360, 900, 385),
        _poly(690, 520, 900, 545),
    ]

    _, metrics = _column_aware_order(left + right, 1000, 1200)

    assert metrics["layout_mode"] == "signature_block"
    assert metrics["column_left_counts"] == [10]
    assert metrics["column_right_counts"] == [3]


def test_signature_block_preserves_body_prefix_before_recipient_columns():
    # Mô phỏng trang 5 của 01-bct: phần thân bài nằm sát khối Nơi nhận nên
    # cùng một vertical block. Một dòng thân bài có tâm nằm bên phải gutter;
    # 1.3.1 đã đẩy dòng này xuống sau toàn bộ cột trái.
    body = [
        _poly(100, 180, 900, 205),
        _poly(100, 215, 520, 240),
        _poly(500, 250, 980, 275),
    ]
    recipients = [
        _poly(100, 290 + i * 35, 520, 315 + i * 35)
        for i in range(10)
    ]
    signature = [
        _poly(650, 300, 900, 325),
        _poly(690, 510, 900, 535),
    ]

    ordered, metrics = _column_aware_order(
        body + recipients + signature,
        page_width=1000,
        page_height=1200,
    )

    assert ordered[: len(body)] == body
    assert ordered[len(body) : len(body) + len(recipients)] == recipients
    assert ordered[-len(signature) :] == signature
    assert metrics["layout_mode"] == "signature_block"
    assert metrics["signature_boundary_y"] == [300.0]
    assert metrics["signature_prefix_count"] == [3]


def test_normal_columns_do_not_report_signature_boundary():
    left = [_poly(100, 200 + i * 35, 420, 225 + i * 35) for i in range(6)]
    right = [_poly(620, 217 + i * 35, 920, 242 + i * 35) for i in range(6)]

    _, metrics = _column_aware_order(left + right, 1000, 1200)

    assert metrics["layout_mode"] == "column_aware"
    assert metrics["signature_boundary_y"] == [None]
    assert metrics["signature_prefix_count"] == [0]


def test_signature_crossing_heading_is_affiliated_with_right_column():
    # Geometry gần với trang 9: dòng KT. THỦ TƯỚNG cắt gutter do bbox bị kéo
    # bởi con dấu, trong khi hai anchor còn lại nằm rõ ở cột phải.
    body = [
        _poly(344, 205, 1510, 248),
        _poly(266, 243, 936, 290),
    ]
    recipients = [
        _poly(282, 369 + i * 36, 1050, 399 + i * 36)
        for i in range(16)
    ]
    long_recipient = _poly(282, 477, 1250, 507)
    recipients[3] = long_recipient
    kt_crossing = _poly(850, 369, 1450, 402)
    pho = _poly(1300, 407, 1455, 440)
    signer = _poly(1300, 690, 1460, 730)
    stamp_like = _poly(760, 369, 1592, 715)

    ordered, metrics = _column_aware_order(
        body + recipients + [kt_crossing, pho, signer, stamp_like],
        page_width=1673,
        page_height=2353,
    )

    assert ordered[:2] == body
    recipient_slice = ordered[2 : 2 + len(recipients)]
    assert recipient_slice == recipients
    right_slice = ordered[2 + len(recipients) :]
    assert right_slice[:3] == [kt_crossing, pho, signer]
    assert stamp_like in right_slice
    assert metrics["layout_mode"] == "signature_block"
    assert metrics["signature_crossing_to_right_count"] == [1]


def test_long_recipient_crossing_gutter_stays_in_left_column():
    recipients = [
        _poly(100, 300 + i * 35, 520, 325 + i * 35)
        for i in range(9)
    ]
    long_recipient = _poly(100, 615, 760, 640)
    recipients.append(long_recipient)
    signature = [
        _poly(650, 310, 900, 335),
        _poly(670, 360, 900, 385),
        _poly(690, 520, 900, 545),
    ]

    ordered, metrics = _column_aware_order(
        recipients + signature,
        page_width=1000,
        page_height=1200,
    )

    assert ordered[: len(recipients)] == recipients
    assert ordered[-len(signature) :] == signature
    assert metrics["signature_crossing_to_right_count"] == [0]


def test_text_detection_constructor_retries_without_cpu_threads_only_for_compatibility():
    calls = []

    class LegacyTextDetection:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            if "cpu_threads" in kwargs:
                raise TypeError("unexpected keyword argument 'cpu_threads'")
            self.kwargs = kwargs

    processor, threads_effective = PaddleLineDetector._construct_text_detection(
        LegacyTextDetection,
        {
            "model_name": "det",
            "device": "cpu",
            "enable_mkldnn": True,
            "cpu_threads": 4,
        },
    )

    assert threads_effective is False
    assert processor.kwargs["enable_mkldnn"] is True
    assert len(calls) == 2
    assert "cpu_threads" not in calls[-1]


def test_detector_falls_back_after_known_onednn_predict_failure_and_reuses_processor(
    monkeypatch,
):
    constructor_calls = []

    class TextDetection:
        model_version = "test"

        def __init__(self, **kwargs):
            constructor_calls.append(dict(kwargs))
            self.enable_mkldnn = kwargs["enable_mkldnn"]

        def predict(self, image):
            if self.enable_mkldnn:
                raise NotImplementedError(
                    "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
                    "[pir::ArrayAttribute<pir::DoubleAttribute>] "
                    "(at onednn_instruction.cc:118)"
                )
            return FakeProcessor().predict(image)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=TextDetection))
    detector = PaddleLineDetector(Settings(paddle_enable_mkldnn=True))
    page = PageImage(
        page_index=0,
        image=Image.new("RGB", (200, 100), "white"),
        rotation=0,
    )

    first = detector.detect(page)
    second = detector.detect(page)

    assert len(first) == 1
    assert len(second) == 1
    assert [call["enable_mkldnn"] for call in constructor_calls] == [True, False]
    assert detector.init_metrics["fallback_used"] is True
    assert detector.init_metrics["fallback_phase"] == "predict"
    assert detector.init_metrics["mkldnn_effective"] is False


def test_detector_defaults_to_safe_backend_without_throwaway_onednn_inference(
    monkeypatch,
):
    constructor_calls = []

    class TextDetection:
        model_version = "test"

        def __init__(self, **kwargs):
            constructor_calls.append(dict(kwargs))

        def predict(self, image):
            return FakeProcessor().predict(image)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=TextDetection))
    detector = PaddleLineDetector(Settings())
    page = PageImage(
        page_index=0,
        image=Image.new("RGB", (200, 100), "white"),
        rotation=0,
    )

    polygons = detector.detect(page)

    assert len(polygons) == 1
    assert [call["enable_mkldnn"] for call in constructor_calls] == [False]
    assert detector.init_metrics["fallback_used"] is False


def test_detector_does_not_fallback_after_unrelated_predict_failure(monkeypatch):
    constructor_calls = []

    class TextDetection:
        def __init__(self, **kwargs):
            constructor_calls.append(dict(kwargs))

        def predict(self, image):
            raise RuntimeError("unrelated detector failure")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=TextDetection))
    detector = PaddleLineDetector(Settings(paddle_enable_mkldnn=True))
    page = PageImage(
        page_index=0,
        image=Image.new("RGB", (200, 100), "white"),
        rotation=0,
    )

    with pytest.raises(RuntimeError, match="unrelated detector failure"):
        detector.detect(page)

    assert [call["enable_mkldnn"] for call in constructor_calls] == [True]


def test_detector_records_constructor_fallback_phase(monkeypatch):
    constructor_calls = []

    class TextDetection:
        model_version = "test"

        def __init__(self, **kwargs):
            constructor_calls.append(dict(kwargs))
            if kwargs["enable_mkldnn"]:
                raise RuntimeError("oneDNN initialization failed")

        def predict(self, image):
            return FakeProcessor().predict(image)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=TextDetection))
    detector = PaddleLineDetector(Settings(paddle_enable_mkldnn=True))
    page = PageImage(
        page_index=0,
        image=Image.new("RGB", (200, 100), "white"),
        rotation=0,
    )

    polygons = detector.detect(page)

    assert len(polygons) == 1
    assert [call["enable_mkldnn"] for call in constructor_calls] == [True, False]
    assert detector.init_metrics["fallback_used"] is True
    assert detector.init_metrics["fallback_phase"] == "init"
    assert detector.init_metrics["mkldnn_effective"] is False


def test_detector_retries_known_onednn_predict_failure_only_once(monkeypatch):
    constructor_calls = []

    class TextDetection:
        def __init__(self, **kwargs):
            constructor_calls.append(dict(kwargs))
            self.enable_mkldnn = kwargs["enable_mkldnn"]

        def predict(self, image):
            if self.enable_mkldnn:
                raise NotImplementedError(
                    "ConvertPirAttribute2RuntimeAttribute failed at onednn_instruction.cc"
                )
            raise RuntimeError("safe backend failed")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=TextDetection))
    detector = PaddleLineDetector(Settings(paddle_enable_mkldnn=True))
    page = PageImage(
        page_index=0,
        image=Image.new("RGB", (200, 100), "white"),
        rotation=0,
    )

    with pytest.raises(RuntimeError, match="safe backend failed"):
        detector.detect(page)

    assert [call["enable_mkldnn"] for call in constructor_calls] == [True, False]
