import threading
from types import SimpleNamespace

from government_ocr_text_api.config import Settings
from government_ocr_text_api.models import Recognition
from government_ocr_text_api.ocr_pipeline import OcrPagePipeline, PreparedOcrPage
from government_ocr_text_api.quality import PageQualityResult


def test_quality_risk_line_is_kept_and_page_is_marked_for_review():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    prepared = PreparedOcrPage(
        page_index=0,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[],
        metrics={},
    )
    result = pipeline._finalize(
        prepared,
        [
            Recognition(
                text="Nội dung còn đọc được",
                confidence=0.9,
                error_code="width_cap_unresolved",
                message_vi="Cần đối chiếu",
            )
        ],
    )
    assert result.text == "Nội dung còn đọc được"
    assert result.needs_review is True
    assert result.error_codes == ["width_cap_unresolved"]


def test_finalize_exposes_independent_verifier_evidence():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    prepared = PreparedOcrPage(
        page_index=0,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[],
        metrics={},
    )

    result = pipeline._finalize(
        prepared,
        [
            Recognition(
                text="Cơ quan có thẩm quyền",
                confidence=0.96,
                semantic_risk="high",
                semantic_reasons=("tesseract_diacritic_disagreement",),
                verifier_text="Cơ quan có thẩm quyên",
                verifier_confidence=0.95,
            )
        ],
    )

    assert result.line_results[0].verifier_text == "Cơ quan có thẩm quyên"
    assert result.line_results[0].verifier_confidence == 0.95


class SelectiveSemanticRecognizer:
    def __init__(self):
        self.calls = []

    def verify_semantic_candidates(
        self,
        crops,
        recognitions,
        candidate_indices,
        page_metrics,
        retry_crops=None,
    ):
        self.calls.append(list(candidate_indices))
        assert retry_crops is not None
        assert len(retry_crops) == len(crops)
        page_metrics[0]["semantic_candidate_count"] = len(candidate_indices)
        return list(recognitions)


class WindowSelectiveSemanticRecognizer:
    def __init__(self):
        self.calls = []

    def verify_semantic_candidates(
        self,
        crops,
        recognitions,
        candidate_indices,
        page_metrics,
        retry_crops=None,
    ):
        self.calls.append(
            {
                "crop_ids": [crop.crop_id for crop in crops],
                "candidate_indices": list(candidate_indices),
                "metric_pages": sorted(page_metrics),
                "retry_crop_ids": [crop.crop_id for crop in retry_crops or ()],
            }
        )
        return list(recognitions)


def test_selective_semantic_verifier_routes_only_risky_lines_after_tesseract():
    recognizer = SelectiveSemanticRecognizer()
    pipeline = OcrPagePipeline(
        Settings(semantic_selective_verification_enabled=True),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    page = _prepared_chapter_page()
    page.crops.append(
        _line_crop(
            "p0000-l0001",
            300,
            260,
            470,
            290,
            Image.new("RGB", (180, 36), "white"),
        )
    )
    recognitions = [
        Recognition(
            "Điều 14",
            0.96,
            semantic_risk="high",
            semantic_reasons=("tesseract_numeric_disagreement",),
            verifier_text="Điều 44",
            verifier_confidence=0.97,
        ),
        Recognition("Nội dung đã được hai máy đồng thuận", 0.95),
    ]
    metrics = {}

    result = pipeline._run_selective_semantic_verification(
        page,
        recognitions,
        metrics,
    )

    assert recognizer.calls == [[0]]
    assert result == recognitions
    assert metrics["semantic_candidate_count"] == 1


def test_selective_semantic_verifier_skips_lines_already_checked_by_secondary():
    recognizer = SelectiveSemanticRecognizer()
    pipeline = OcrPagePipeline(
        Settings(semantic_selective_verification_enabled=True),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    page = _prepared_chapter_page()
    recognition = Recognition(
        "Chương III",
        0.90,
        semantic_risk="high",
        semantic_reasons=("tesseract_material_disagreement",),
        secondary_confidence=0.98,
    )

    result = pipeline._run_selective_semantic_verification(
        page,
        [recognition],
        {},
    )

    assert recognizer.calls == []
    assert result == [recognition]


def test_selective_semantic_verifier_always_checks_legal_numbers():
    recognizer = SelectiveSemanticRecognizer()
    pipeline = OcrPagePipeline(
        Settings(semantic_selective_verification_enabled=True),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    page = _prepared_chapter_page()
    recognition = Recognition(
        "Luật số 33/2009/QH12",
        0.97,
        verifier_text="Luật số 33/2009/QH12",
        verifier_confidence=0.98,
    )

    pipeline._run_selective_semantic_verification(page, [recognition], {})

    assert recognizer.calls == [[0]]


def test_selective_semantic_verifier_checks_punctuation_only_disagreement():
    recognizer = SelectiveSemanticRecognizer()
    pipeline = OcrPagePipeline(
        Settings(semantic_selective_verification_enabled=True),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    page = _prepared_chapter_page()
    recognition = Recognition(
        "Giám đốc Sở Xây dựng, Giám đốc Sở Y tế;",
        0.97,
        verifier_text="Giám đốc Sở Xây dựng; Giám đốc Sở Y tế;",
        verifier_confidence=0.98,
    )

    pipeline._run_selective_semantic_verification(page, [recognition], {})

    assert recognizer.calls == [[0]]


def test_selective_semantic_verifier_batches_candidates_across_page_window():
    recognizer = WindowSelectiveSemanticRecognizer()
    pipeline = OcrPagePipeline(
        Settings(semantic_selective_verification_enabled=True),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    first = _prepared_chapter_page()
    second = _prepared_chapter_page()
    second.page_index = 1
    second.page_image.page_index = 1
    second.crops[0].crop_id = "p0001-l0000"
    recognitions = {
        0: [Recognition("Luật số 33/2009/QH12", 0.97)],
        1: [
            Recognition(
                "Giám đốc Sở Xây dựng, Giám đốc Sở Y tế;",
                0.97,
                verifier_text="Giám đốc Sở Xây dựng; Giám đốc Sở Y tế;",
                verifier_confidence=0.98,
            )
        ],
    }
    metrics = {0: {}, 1: {}}

    result = pipeline._run_selective_semantic_verification_window(
        [first, second],
        recognitions,
        metrics,
    )

    assert len(recognizer.calls) == 1
    assert recognizer.calls[0] == {
        "crop_ids": ["p0000-l0000", "p0001-l0000"],
        "retry_crop_ids": ["p0000-l0000", "p0001-l0000"],
        "candidate_indices": [0, 1],
        "metric_pages": [0, 1],
    }
    assert result == recognitions


def test_page_tesseract_verification_overlaps_primary_recognition(monkeypatch):
    verifier_started = threading.Event()

    class OverlapRecognizer:
        last_page_metrics = {0: {"recognition_ms_allocated": 0.0}}

        def __init__(self):
            self.overlapped = False

        def recognize(self, crops):
            self.overlapped = verifier_started.wait(timeout=0.25)
            return [Recognition("Nội dung", 0.95) for _ in crops]

    class OverlapVerifier:
        def collect_page(self, page_image):
            verifier_started.set()
            return [], {"status": "complete", "elapsed_ms": 0.0}

        def verify_collected(self, crops, recognitions, lines, metrics):
            return list(recognitions), {"elapsed_ms": 0.0}

    recognizer = OverlapRecognizer()
    pipeline = OcrPagePipeline(
        Settings(
            chapter_heading_retry_enabled=False,
            semantic_selective_verification_enabled=False,
            partial_remediation_enabled=False,
        ),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
        tesseract_verifier=OverlapVerifier(),
    )
    page = _prepared_chapter_page()
    monkeypatch.setattr(pipeline, "prepare_page", lambda *_: page)

    result = pipeline.extract_pages(object(), [(0, "scan")])

    assert recognizer.overlapped is True
    assert result[0].text == "Nội dung"


from PIL import Image

from government_ocr_text_api.models import LineCrop, LinePolygon, PageImage


class ChapterRetryRecognizer:
    def __init__(self, retry: Recognition):
        self.retry = retry
        self.calls = 0

    def recognize_targeted(self, crops):
        self.calls += len(crops)
        return [self.retry for _ in crops]


def _prepared_chapter_page() -> PreparedOcrPage:
    page_image = PageImage(
        page_index=0,
        image=Image.new("RGB", (1000, 1200), "white"),
        rotation=0,
    )
    polygon = LinePolygon(
        [(300, 200), (470, 200), (470, 230), (300, 230)],
        0.98,
        "test",
        "test",
    )
    crop = LineCrop(
        "p0000-l0000",
        Image.new("RGB", (180, 36), "white"),
        polygon,
    )
    return PreparedOcrPage(
        page_index=0,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[crop],
        metrics={},
        page_image=page_image,
    )


def test_chapter_heading_retry_accepts_only_complete_heading_pattern():
    recognizer = ChapterRetryRecognizer(Recognition("Chương III", 0.82))
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    metrics = {}

    result = pipeline._refine_chapter_headings(
        _prepared_chapter_page(),
        [Recognition("Chương", 0.88)],
        metrics,
    )

    assert result[0].text == "Chương III"
    assert result[0].error_code is None
    assert recognizer.calls == 1
    assert metrics["chapter_heading_retry_accepted_count"] == 1
    assert metrics["chapter_heading_retry_unresolved_count"] == 0


def test_chapter_heading_retry_rejects_noisy_expansion_and_flags_review():
    recognizer = ChapterRetryRecognizer(
        Recognition("Chương người thuy nhiễu", 0.95)
    )
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    metrics = {}

    result = pipeline._refine_chapter_headings(
        _prepared_chapter_page(),
        [Recognition("Chương", 0.88)],
        metrics,
    )

    assert result[0].text == "Chương"
    assert result[0].error_code == "chapter_heading_incomplete"
    assert result[0].semantic_risk == "high"
    assert "chapter_heading_incomplete" in result[0].semantic_reasons
    assert metrics["chapter_heading_retry_accepted_count"] == 0
    assert metrics["chapter_heading_retry_unresolved_count"] == 1

    finalized = pipeline._finalize(_prepared_chapter_page(), result)
    assert "OCR_SEMANTIC_RISK page=1 line=1" in finalized.ai_safe_text
    assert finalized.ai_ready is False


def test_structural_heading_retry_label_is_configurable_not_corpus_locked():
    recognizer = ChapterRetryRecognizer(Recognition("Phần XII", 0.84))
    pipeline = OcrPagePipeline(
        Settings(chapter_heading_retry_labels="phần"),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
    )
    metrics = {}

    result = pipeline._refine_chapter_headings(
        _prepared_chapter_page(),
        [Recognition("Phần", 0.86)],
        metrics,
    )

    assert result[0].text == "Phần XII"
    assert metrics["chapter_heading_retry_accepted_count"] == 1


def _line_crop(crop_id: str, x0: int, y0: int, x1: int, y1: int, image: Image.Image) -> LineCrop:
    polygon = LinePolygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        0.99,
        "test",
        "test",
    )
    return LineCrop(crop_id, image, polygon)


def test_finalize_keeps_equal_text_on_distinct_geometric_lines():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    page = PreparedOcrPage(
        page_index=0,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[
            _line_crop("p0000-l0000", 10, 10, 200, 35, Image.new("RGB", (190, 25), "white")),
            _line_crop("p0000-l0001", 10, 60, 200, 85, Image.new("RGB", (190, 25), "white")),
        ],
        metrics={},
    )

    result = pipeline._finalize(
        page,
        [Recognition("Dòng lặp hợp lệ", 0.95), Recognition("Dòng lặp hợp lệ", 0.94)],
    )

    assert result.text == "Dòng lặp hợp lệ\nDòng lặp hợp lệ"
    assert result.line_count == 2
    assert result.metrics["geometry_duplicate_suppressed_count"] == 0


def test_finalize_suppresses_equal_text_only_when_geometry_overlaps():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    page = PreparedOcrPage(
        page_index=0,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[
            _line_crop("p0000-l0000", 10, 10, 200, 35, Image.new("RGB", (190, 25), "white")),
            _line_crop("p0000-l0001", 12, 11, 202, 36, Image.new("RGB", (190, 25), "white")),
        ],
        metrics={},
    )

    result = pipeline._finalize(
        page,
        [Recognition("Một dòng", 0.95), Recognition("Một dòng", 0.94)],
    )

    assert result.text == "Một dòng"
    assert result.line_count == 1
    assert result.metrics["geometry_duplicate_suppressed_count"] == 1


def test_geometry_dedup_merges_semantic_risk_independent_of_input_order():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )

    def page():
        return PreparedOcrPage(
            page_index=0,
            ocr_reason="scanned",
            quality=PageQualityResult("keep_original", 0.0, 0.1, False),
            crops=[
                _line_crop("p0000-l0000", 10, 10, 300, 35, Image.new("RGB", (290, 25), "white")),
                _line_crop("p0000-l0001", 12, 11, 302, 36, Image.new("RGB", (290, 25), "white")),
            ],
            metrics={},
        )

    safe = Recognition("Cùng một dòng", 0.96)
    risky = Recognition(
        "Cùng một dòng",
        0.80,
        semantic_risk="high",
        semantic_reasons=("secondary_material_disagreement",),
        secondary_confidence=0.98,
    )

    safe_first = pipeline._finalize(page(), [safe, risky])
    risky_first = pipeline._finalize(page(), [risky, safe])

    for result in (safe_first, risky_first):
        assert result.text == "Cùng một dòng"
        assert result.ai_ready is False
        assert result.semantic_risk_count == 1
        assert "OCR_SEMANTIC_RISK page=1 line=1" in result.ai_safe_text
        assert result.line_results[0].semantic_risk == "high"
        assert result.line_results[0].semantic_reasons == [
            "secondary_material_disagreement"
        ]
        assert result.metrics["geometry_duplicate_suppressed_count"] == 1


def test_nontext_crop_filter_rejects_wide_horizontal_rule_but_keeps_glyph_band():
    from government_ocr_text_api.ocr_pipeline import _nontext_crop_reason

    settings = Settings()
    rule = Image.new("RGB", (400, 32), "white")
    rule_pixels = rule.load()
    for x in range(20, 380):
        for y in range(15, 17):
            rule_pixels[x, y] = (0, 0, 0)
    rule_crop = _line_crop("p0000-l0000", 0, 0, 400, 32, rule)

    text_like = Image.new("RGB", (400, 32), "white")
    text_pixels = text_like.load()
    for x in range(20, 380, 45):
        for px in range(x, min(x + 24, 390)):
            for y in range(7, 26):
                text_pixels[px, y] = (0, 0, 0)
    text_crop = _line_crop("p0000-l0001", 0, 0, 400, 32, text_like)

    assert _nontext_crop_reason(rule_crop, settings) == "horizontal_rule_or_dotted_leader"
    assert _nontext_crop_reason(text_crop, settings) is None


def test_nontext_filter_keeps_dotted_component_rich_band():
    from government_ocr_text_api.ocr_pipeline import _nontext_crop_reason

    dotted = Image.new("RGB", (400, 32), "white")
    pixels = dotted.load()
    for x in range(20, 380, 12):
        for px in range(x, x + 3):
            for y in range(15, 18):
                pixels[px, y] = (0, 0, 0)
    crop = _line_crop("p0000-l0000", 0, 0, 400, 32, dotted)

    assert _nontext_crop_reason(crop, Settings()) is None


def test_nontext_filter_keeps_padded_component_rich_legal_text():
    from government_ocr_text_api.ocr_pipeline import _nontext_crop_reason

    image = Image.new("RGB", (720, 52), "white")
    pixels = image.load()
    for x in range(20, 700, 30):
        for px in range(x, min(x + 20, 710)):
            for y in range(17, 36):
                pixels[px, y] = (0, 0, 0)
    crop = _line_crop("p0000-l0000", 0, 0, 720, 52, image)

    assert _nontext_crop_reason(crop, Settings()) is None


def test_axis_aligned_semantic_retry_crop_expands_all_sides_without_perspective_warp():
    from government_ocr_text_api.line_crops import make_axis_aligned_retry_crop

    page = PageImage(page_index=0, image=Image.new("RGB", (100, 100), "white"))
    original = _line_crop(
        "p0000-l0000",
        20,
        30,
        80,
        50,
        Image.new("RGB", (60, 20), "white"),
    )

    retry = make_axis_aligned_retry_crop(
        page,
        original,
        padding_height_ratio=0.20,
    )

    assert retry.crop_id == original.crop_id
    assert retry.image.size == (68, 28)
    assert retry.polygon.bbox.x0 == 16.0
    assert retry.polygon.bbox.y0 == 26.0
    assert retry.polygon.bbox.x1 == 84.0
    assert retry.polygon.bbox.y1 == 54.0
    assert retry.polygon.source == "semantic_retry_axis_aligned"


def test_finalize_masks_high_risk_line_from_ai_safe_text():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    page = PreparedOcrPage(
        page_index=0,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[
            _line_crop("p0000-l0000", 10, 10, 300, 35, Image.new("RGB", (290, 25), "white")),
            _line_crop("p0000-l0001", 10, 50, 500, 75, Image.new("RGB", (490, 25), "white")),
        ],
        metrics={},
    )

    result = pipeline._finalize(
        page,
        [
            Recognition("Dòng an toàn", 0.96),
            Recognition(
                "xâm phạm ninh quốc gia, khủng bố",
                0.85,
                semantic_risk="high",
                semantic_reasons=("secondary_indicates_primary_omission",),
                secondary_confidence=0.98,
            ),
        ],
    )

    assert result.text == "Dòng an toàn\nxâm phạm ninh quốc gia, khủng bố"
    assert result.ai_safe_text == (
        "Dòng an toàn\n"
        "[OCR_SEMANTIC_RISK page=1 line=2 "
        "reasons=secondary_indicates_primary_omission]"
    )
    assert result.ai_ready is False
    assert result.semantic_risk_count == 1
    assert result.line_results[1].text == "xâm phạm ninh quốc gia, khủng bố"
    assert result.line_results[1].semantic_risk == "high"
    assert result.line_results[1].bbox == [10.0, 50.0, 500.0, 75.0]


def test_partial_remediation_prioritizes_numeric_risk_and_runs_only_once():
    from government_ocr_text_api.models import BBox
    from government_ocr_text_api.tesseract_verifier import TesseractLine

    class TargetedVerifier:
        def __init__(self):
            self.calls = []

        def recognize_targeted(self, crops):
            self.calls.append([crop.crop_id for crop in crops])
            return (
                [TesseractLine("Điều 15 có hiệu lực", 0.97, crops[0].polygon.bbox)],
                {
                    "status": "complete",
                    "attempted_count": 1,
                    "matched_count": 1,
                    "elapsed_ms": 12.0,
                    "events": [{"crop_id": crops[0].crop_id, "psm": 7}],
                },
            )

    class RemediationRecognizer:
        def __init__(self):
            self.calls = []

        def remediate_high_risk_candidates(
            self,
            crops,
            recognitions,
            page_metrics,
        ):
            self.calls.append(
                {
                    "crop_ids": [crop.crop_id for crop in crops],
                    "verifier_texts": [item.verifier_text for item in recognitions],
                }
            )
            page_metrics[0].update(
                attempted_count=1,
                applied_count=1,
                elapsed_ms=20.0,
                events=[{"crop_id": crops[0].crop_id, "applied": True}],
            )
            return [
                Recognition(
                    "Điều 15 có hiệu lực",
                    0.91,
                    raw_text=recognitions[0].text,
                    semantic_risk="medium",
                    semantic_reasons=("partial_remediation_consensus_applied",),
                    verifier_text=recognitions[0].verifier_text,
                    verifier_confidence=recognitions[0].verifier_confidence,
                )
            ]

    page = _prepared_chapter_page()
    page.crops = [
        _line_crop("p0000-l0000", 10, 10, 400, 35, Image.new("RGB", (390, 25), "white")),
        _line_crop("p0000-l0001", 10, 50, 400, 75, Image.new("RGB", (390, 25), "white")),
        _line_crop("p0000-l0002", 10, 90, 400, 115, Image.new("RGB", (390, 25), "white")),
    ]
    recognitions = {
        0: [
            Recognition(
                "Nội dung có thể thiếu từ",
                0.82,
                semantic_risk="high",
                semantic_reasons=("tesseract_material_disagreement",),
            ),
            Recognition(
                "Điều 16 có hiệu lực",
                0.90,
                semantic_risk="high",
                semantic_reasons=("tesseract_numeric_disagreement",),
            ),
            Recognition("Dòng đã an toàn", 0.96),
        ]
    }
    verifier = TargetedVerifier()
    recognizer = RemediationRecognizer()
    pipeline = OcrPagePipeline(
        Settings(
            partial_remediation_enabled=True,
            partial_remediation_max_lines_per_page=1,
        ),
        detector=SimpleNamespace(),
        recognizer=recognizer,
        quality_gate=SimpleNamespace(),
        tesseract_verifier=verifier,
    )
    metrics = {0: {}}

    result = pipeline._run_partial_remediation_window([page], recognitions, metrics)

    assert verifier.calls == [["p0000-l0001"]]
    assert recognizer.calls == [
        {
            "crop_ids": ["p0000-l0001"],
            "verifier_texts": ["Điều 15 có hiệu lực"],
        }
    ]
    assert result[0][0].semantic_risk == "high"
    assert result[0][1].text == "Điều 15 có hiệu lực"
    assert result[0][1].semantic_risk == "medium"
    assert result[0][2].text == "Dòng đã an toàn"
    assert metrics[0]["semantic_high_risk_count"] == 1
    assert page.metrics["partial_remediation"]["candidate_count"] == 2
    assert page.metrics["partial_remediation"]["selected_count"] == 1


def test_finalize_uses_validated_medium_risk_text_and_retains_raw_text():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    page = PreparedOcrPage(
        page_index=1,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[
            _line_crop("p0001-l0000", 20, 20, 420, 50, Image.new("RGB", (400, 30), "white")),
        ],
        metrics={},
    )
    raw = "Nội dung đúng người thuy nhiễu"

    result = pipeline._finalize(
        page,
        [
            Recognition(
                "Nội dung đúng",
                0.60,
                error_code="decoder_loop_trimmed",
                raw_text=raw,
                semantic_risk="medium",
                semantic_reasons=("unsupported_suffix_removed",),
                secondary_confidence=0.97,
            )
        ],
    )

    assert result.text == "Nội dung đúng"
    assert result.ai_safe_text == "Nội dung đúng"
    assert result.ai_ready is True
    assert result.line_results[0].raw_text == raw


def test_finalize_emits_placeholder_for_empty_failed_recognition():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    page = PreparedOcrPage(
        page_index=2,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[
            _line_crop("p0002-l0000", 0, 0, 300, 30, Image.new("RGB", (300, 30), "white")),
        ],
        metrics={},
    )

    result = pipeline._finalize(
        page,
        [
            Recognition(
                "",
                0.0,
                error_code="recognition_failed",
                semantic_risk="high",
                semantic_reasons=("empty_primary_text",),
            )
        ],
    )

    assert result.text == ""
    assert result.ai_safe_text == (
        "[OCR_SEMANTIC_RISK page=3 line=1 reasons=empty_primary_text]"
    )
    assert result.ai_ready is False
    assert len(result.line_results) == 1


def test_finalize_masks_page_when_detector_returns_no_crops():
    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=SimpleNamespace(),
    )
    page = PreparedOcrPage(
        page_index=3,
        ocr_reason="scanned",
        quality=PageQualityResult("keep_original", 0.0, 0.1, False),
        crops=[],
        metrics={},
    )

    result = pipeline._finalize(page, [])

    assert result.text == ""
    assert result.ai_safe_text == (
        "[OCR_SEMANTIC_RISK page=4 line=0 reasons=empty_ocr_text]"
    )
    assert result.ai_ready is False
    assert result.semantic_risk_count == 1
    assert result.error_codes == ["empty_ocr_text", "semantic_risk_detected"]
    assert result.line_results[0].semantic_risk == "high"


def test_low_ink_fast_path_is_not_reported_ai_ready():
    page_image = PageImage(
        page_index=4,
        image=Image.new("RGB", (600, 800), "white"),
    )

    class Session:
        def render_page(self, page_index):
            assert page_index == 4
            return page_image

    class LowInkGate:
        def evaluate(self, page):
            return PageQualityResult("keep_original", 0.0, 0.0, True)

        def apply(self, page, quality):
            return page

    pipeline = OcrPagePipeline(
        Settings(),
        detector=SimpleNamespace(),
        recognizer=SimpleNamespace(),
        quality_gate=LowInkGate(),
    )

    prepared = pipeline.prepare_page(Session(), 4, "pdf_inspector_requires_ocr")
    result = prepared.early_result

    assert result is not None
    assert result.ai_safe_text == (
        "[OCR_SEMANTIC_RISK page=5 line=0 reasons=empty_ocr_text,low_ink_page]"
    )
    assert result.ai_ready is False
    assert result.semantic_risk_count == 1
    assert result.needs_review is True
    assert result.error_codes == ["empty_ocr_text"]
