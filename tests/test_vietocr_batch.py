from PIL import Image

from government_ocr_text_api.config import Settings
from government_ocr_text_api.models import LineCrop, LinePolygon, Recognition
from government_ocr_text_api.vietocr_recognizer import (
    VietOcrRecognizer,
    _find_character_loop,
    _find_decoder_loop,
    _prioritize_semantic_retry_indices,
    _trim_character_loop,
    _trim_decoder_loop,
)


def _synthetic_text_line(width: int, height: int = 32) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()
    word_width = 120
    gap = 28
    x = 8
    while x < width - 8:
        right = min(width - 8, x + word_width)
        for px in range(x, right):
            for py in range(8, height - 7):
                pixels[px, py] = (0, 0, 0)
        x = right + gap
    return image


class Predictor:
    config = {
        "dataset": {"image_height": 32, "image_max_width": 512},
        "predictor": {"beamsearch": False},
    }

    def predict_batch(self, images, return_prob=True):
        return ["Xin chào"] * len(images), [0.95] * len(images)


def test_batch_recognition_keeps_order():
    polygon = LinePolygon([(0, 0), (20, 0), (20, 10), (0, 10)], 1.0, "test", "test")
    crops = [LineCrop(str(i), Image.new("RGB", (100 + i, 32), "white"), polygon) for i in range(3)]
    recognizer = VietOcrRecognizer(Settings(recognition_batch_size=2), predictor=Predictor())
    result = recognizer.recognize(crops)
    assert [item.text for item in result] == ["Xin chào"] * 3
    assert recognizer.last_batch_metrics["batch_count"] == 2


def test_safe_weights_wrapper_passes_weights_only(monkeypatch):
    import sys
    from types import SimpleNamespace

    from government_ocr_text_api.vietocr_recognizer import _build_predictor_with_safe_weights

    seen = {}

    def fake_load(*args, **kwargs):
        seen.update(kwargs)
        return {"weight": 1}

    fake_torch = SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self, config):
            import torch
            self.state = torch.load("weights.pth", map_location="cpu")

    predictor = _build_predictor_with_safe_weights(FakePredictor, {}, True)
    assert predictor.state == {"weight": 1}
    assert seen["weights_only"] is True
    assert fake_torch.load is fake_load


class RecordingPredictor:
    config = {
        "dataset": {"image_height": 32, "image_max_width": 512},
        "predictor": {"beamsearch": False},
    }

    def __init__(self):
        self.batch_sizes = []
        self.batch_resized_widths = []
        self.predict_calls = 0

    def predict_batch(self, images, return_prob=True):
        self.batch_sizes.append(len(images))
        self.batch_resized_widths.append(
            {
                ((32 * image.width / max(image.height, 1) + 9) // 10) * 10
                for image in images
            }
        )
        return ["đoạn văn"] * len(images), [0.95] * len(images)

    def predict(self, image, return_prob=False):
        self.predict_calls += 1
        return "Nội dung đúng hoàn chỉnh"


def test_wide_crop_is_split_before_vietocr_width_cap():
    polygon = LinePolygon([(0, 0), (1800, 0), (1800, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", _synthetic_text_line(1800), polygon)
    predictor = RecordingPredictor()
    recognizer = VietOcrRecognizer(Settings(), predictor=predictor)

    result = recognizer.recognize([crop])

    assert result[0].text
    assert recognizer.last_batch_metrics["wide_crop_count"] == 1
    assert recognizer.last_batch_metrics["segment_count"] >= 3
    assert recognizer.last_batch_metrics["max_resized_width_after_split"] <= 512
    assert recognizer.last_batch_metrics["width_capped_after_split_count"] == 0


def test_width_capped_crop_without_safe_word_gap_is_flagged_for_review():
    polygon = LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test")
    # Một dải mực liên tục mô phỏng trường hợp không có khoảng trắng an toàn.
    image = Image.new("RGB", (900, 32), "black")
    crop = LineCrop("p0000-l0000", image, polygon)
    predictor = RecordingPredictor()
    recognizer = VietOcrRecognizer(Settings(), predictor=predictor)

    result = recognizer.recognize([crop])

    assert recognizer.last_batch_metrics["wide_crop_count"] == 0
    assert recognizer.last_batch_metrics["width_cap_detected_count"] == 1
    assert recognizer.last_batch_metrics["width_cap_unresolved_count"] == 1
    assert result[0].error_code == "width_cap_unresolved"


class LoopPredictor(RecordingPredictor):
    def predict_batch(self, images, return_prob=True):
        self.batch_sizes.append(len(images))
        return (
            ["Nội dung đúng nhiều thuy nhiều thuy nhiều thuy"] * len(images),
            [0.40] * len(images),
        )


def test_beam_retry_is_limited_per_page_and_remaining_loops_are_trimmed():
    polygon = LinePolygon([(0, 0), (300, 0), (300, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(f"p0000-l{i:04d}", Image.new("RGB", (300, 32), "white"), polygon)
        for i in range(5)
    ]
    predictor = LoopPredictor()
    recognizer = VietOcrRecognizer(
        Settings(max_beam_retries_per_page=2),
        predictor=predictor,
    )

    result = recognizer.recognize(crops)

    assert predictor.predict_calls == 2
    assert recognizer.last_batch_metrics["beam_retry_count"] == 2
    assert recognizer.last_batch_metrics["decoder_loop_detected_count"] == 5
    assert all("nhiều thuy nhiều thuy nhiều thuy" not in item.text for item in result)


def test_adaptive_batching_uses_pixel_budget_across_pages():
    polygon = LinePolygon([(0, 0), (300, 0), (300, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(
            f"p{page:04d}-l{line:04d}",
            Image.new("RGB", (300, 32), "white"),
            polygon,
        )
        for page in range(2)
        for line in range(9)
    ]
    predictor = RecordingPredictor()
    recognizer = VietOcrRecognizer(
        Settings(recognition_pixel_budget=2400, recognition_batch_size=32),
        predictor=predictor,
    )

    recognizer.recognize(crops)

    assert predictor.batch_sizes == [8, 8, 2]
    assert all(len(widths) == 1 for widths in predictor.batch_resized_widths)
    assert recognizer.last_batch_metrics["internal_batch_count_estimate"] == 3
    assert recognizer.last_batch_metrics["page_count"] == 2
    assert set(recognizer.last_page_metrics) == {0, 1}

def test_bigram_loop_is_detected_after_three_repeats():
    text = "Nội dung hợp lệ người thay người thay người thay điện tử"
    issue = _find_decoder_loop(text)
    assert issue is not None
    assert issue.ngram_size == 2
    assert _trim_decoder_loop(text, issue) == "Nội dung hợp lệ"


def test_decoder_loop_never_trims_a_numeric_removed_span():
    text = "Dieu khoan 2025 2025 2025"
    issue = _find_decoder_loop(text)

    assert issue is not None
    assert _trim_decoder_loop(text, issue) == text


def test_numeric_decoder_loop_is_preserved_and_marked_high_risk():
    class NumericLoopPredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            return ["Dieu khoan 2025 2025 2025"] * len(images), [0.95] * len(images)

    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(500),
        LinePolygon([(0, 0), (500, 0), (500, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=False,
            split_wide_crops=False,
            decoder_evidence_enabled=False,
            beam_retry_enabled=False,
            tail_segment_validation=False,
        ),
        predictor=NumericLoopPredictor(),
    )

    result = recognizer.recognize([crop])

    assert result[0].text == "Dieu khoan 2025 2025 2025"
    assert result[0].error_code == "numeric_ocr_pattern_unresolved"
    assert result[0].semantic_risk == "high"


def test_two_bigram_repeats_can_be_legitimate():
    text = "Văn phòng đại diện/đại diện theo ủy quyền"
    assert _find_decoder_loop(text) is None


def test_loop_with_only_one_prefix_token_is_dropped():
    text = "linh thu thu thu thu thu"
    issue = _find_decoder_loop(text)
    assert issue is not None
    assert _trim_decoder_loop(text, issue) == ""


def test_repeated_character_run_is_trimmed():
    text = "Kính chuyển: 1991.000000000000000000"
    start = _find_character_loop(text)
    assert start is not None
    assert _trim_character_loop(text, start) == text


def test_repeated_punctuation_run_is_trimmed():
    text = "a) Bãi bỏ khoản 2 Điều 1, Điều 3, , , , , , , , ,"
    start = _find_character_loop(text)
    assert start is not None
    assert _trim_character_loop(text, start) == "a) Bãi bỏ khoản 2 Điều 1, Điều 3"



def test_wide_split_adds_pixel_overlap_between_adjacent_segments():
    from government_ocr_text_api.vietocr_recognizer import _split_wide_image

    image = _synthetic_text_line(1500)
    segments = _split_wide_image(
        image,
        image_height=32,
        image_max_width=512,
        target_ratio=0.92,
        max_segments=6,
        valley_max_ink_ratio=0.02,
        overlap_height_ratio=0.55,
    )

    assert len(segments) >= 3
    assert all(item.left_overlap or item.right_overlap for item in segments)
    assert all(
        left.source_right > right.source_left
        for left, right in zip(segments, segments[1:])
    )
    assert all(
        ((32 * item.image.width / item.image.height + 9) // 10) * 10 <= 512
        for item in segments
    )


def test_seam_aware_join_removes_repeated_short_word_at_overlap():
    from government_ocr_text_api.vietocr_recognizer import _join_segments

    text, merge_count = _join_segments(
        ["hoạt động xâm phạm an", "an ninh quốc gia, khủng bố"],
        max_token_overlap=4,
        max_char_overlap=32,
    )

    assert text == "hoạt động xâm phạm an ninh quốc gia, khủng bố"
    assert merge_count == 1


def test_partial_third_ngram_at_terminal_is_detected_and_trimmed():
    text = (
        "Nội dung hợp lệ người thuyên theng thuy nhiệu viện "
        "the then gia nguyên the then gia nguyên the then"
    )
    issue = _find_decoder_loop(text)

    assert issue is not None
    assert issue.ngram_size == 4
    assert issue.repeats == 2
    assert issue.partial_tokens == 2
    assert _trim_decoder_loop(text, issue).endswith("nhiệu viện")


def test_two_complete_ngrams_without_partial_tail_are_not_trimmed():
    text = "Nội dung hợp lệ the then gia nguyên the then gia nguyên"
    assert _find_decoder_loop(text) is None


class BlankTailPredictor(RecordingPredictor):
    def predict_batch(self, images, return_prob=True):
        self.batch_sizes.append(len(images))
        return ["nội dung đúng nam"] * len(images), [0.50] * len(images)


def test_blank_tail_segment_is_marked_uncertain_without_unvalidated_retry():
    image = Image.new("RGB", (1200, 32), "white")
    pixels = image.load()
    for x in range(10, 820):
        if x % 150 < 100:
            for y in range(8, 25):
                pixels[x, y] = (0, 0, 0)
    polygon = LinePolygon([(0, 0), (1200, 0), (1200, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", image, polygon)
    recognizer = VietOcrRecognizer(Settings(), predictor=BlankTailPredictor())

    result = recognizer.recognize([crop])

    assert result[0].text == "nội dung đúng nam"
    assert result[0].error_code == "tail_segment_uncertain"
    assert recognizer.last_batch_metrics["tail_segment_uncertain_count"] == 1


class NumericTailPredictor(RecordingPredictor):
    def predict_batch(self, images, return_prob=True):
        self.batch_sizes.append(len(images))
        return ["mã 2025"] * len(images), [0.30] * len(images)

    def predict(self, image, return_prob=True):
        self.predict_calls += 1
        return "mã 2026", 0.90


def _blank_numeric_tail_crop():
    image = Image.new("RGB", (1200, 32), "white")
    pixels = image.load()
    for x in range(10, 780):
        if x % 150 < 100:
            for y in range(8, 25):
                pixels[x, y] = (0, 0, 0)
    polygon = LinePolygon(
        [(0, 0), (1200, 0), (1200, 32), (0, 32)],
        1.0,
        "test",
        "test",
    )
    return LineCrop("p0000-l0000", image, polygon)


def test_low_ink_tail_never_suppresses_numeric_text():
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=False,
            tail_segment_retry_enabled=False,
        ),
        predictor=NumericTailPredictor(),
    )

    result = recognizer.recognize([_blank_numeric_tail_crop()])

    assert "2025" in result[0].text
    assert result[0].error_code == "numeric_ocr_pattern_unresolved"
    assert result[0].semantic_risk == "high"


def test_tail_retry_never_replaces_disagreeing_numeric_text():
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=False,
            tail_segment_retry_enabled=True,
        ),
        predictor=NumericTailPredictor(),
    )

    result = recognizer.recognize([_blank_numeric_tail_crop()])

    assert "2025" in result[0].text
    assert "2026" not in result[0].text
    assert result[0].error_code == "numeric_ocr_pattern_unresolved"
    assert result[0].semantic_risk == "high"


def test_131_defaults_disable_unvalidated_output_mutations():
    settings = Settings()
    assert settings.experimental_overlap_split is False
    assert settings.wide_crop_overlap_height_ratio == 0.0
    assert settings.tail_segment_retry_enabled is False


def test_stale_overlap_value_is_ignored_without_experimental_flag():
    polygon = LinePolygon(
        [(0, 0), (1500, 0), (1500, 32), (0, 32)],
        1.0,
        "test",
        "test",
    )
    crop = LineCrop("p0000-l0000", _synthetic_text_line(1500), polygon)
    recognizer = VietOcrRecognizer(
        Settings(
            experimental_overlap_split=False,
            wide_crop_overlap_height_ratio=0.55,
        ),
        predictor=RecordingPredictor(),
    )

    recognizer.recognize([crop])

    assert recognizer.last_batch_metrics["split_overlap_source_pixels"] == 0


def test_delete_only_consensus_does_not_mutate_midline_insertions():
    from government_ocr_text_api.vietocr_recognizer import _consensus_delete_only

    decision = _consensus_delete_only(
        "trao đổi, cung xyzzy cấp cho Bộ Công an",
        "trao đổi, cung cấp cho Bộ Công an",
        original_confidence=0.50,
        retry_confidence=0.72,
        leading_blank_ratio=0.0,
        trailing_blank_ratio=0.0,
        width_reduction_ratio=0.20,
        settings=Settings(),
    )

    assert decision is None


def test_delete_only_consensus_never_rewrites_substituted_content():
    from government_ocr_text_api.vietocr_recognizer import _consensus_delete_only

    decision = _consensus_delete_only(
        "Bộ Công an cung cấp thông tin",
        "Bộ Quốc phòng cung cấp thông tin",
        original_confidence=0.50,
        retry_confidence=0.90,
        leading_blank_ratio=0.0,
        trailing_blank_ratio=0.0,
        width_reduction_ratio=0.30,
        settings=Settings(),
    )

    assert decision is None


def test_delete_only_consensus_suppresses_numeric_suffix_only_with_strong_visual_change():
    from government_ocr_text_api.vietocr_recognizer import _consensus_delete_only

    settings = Settings(hallucination_guard_numeric_apply_changes=True)
    rejected = _consensus_delete_only(
        "có hiệu lực năm 2022.000",
        "có hiệu lực năm 2022",
        original_confidence=0.60,
        retry_confidence=0.80,
        leading_blank_ratio=0.0,
        trailing_blank_ratio=0.30,
        width_reduction_ratio=0.08,
        settings=settings,
    )
    accepted = _consensus_delete_only(
        "có hiệu lực năm 2022.000",
        "có hiệu lực năm 2022",
        original_confidence=0.60,
        retry_confidence=0.80,
        leading_blank_ratio=0.0,
        trailing_blank_ratio=0.30,
        width_reduction_ratio=0.25,
        settings=settings,
    )

    assert rejected is None
    assert accepted is not None
    assert accepted.text == "có hiệu lực năm 2022.000"
    assert accepted.numeric_only_warning is True


class ConsensusPredictor(RecordingPredictor):
    def predict_batch(self, images, return_prob=True):
        self.batch_sizes.append(len(images))
        return ["trao đổi, cung cấp xyzzy"] * len(images), [0.50] * len(images)

    def predict(self, image, return_prob=True):
        self.predict_calls += 1
        return "trao đổi, cung cấp", 0.75


def test_hallucination_guard_applies_visual_consensus_and_marks_review():
    image = Image.new("RGB", (700, 32), "white")
    pixels = image.load()
    # Dòng thật chỉ chiếm phần đầu, phần trắng phải tạo retry view khác biệt rõ.
    for x in range(10, 470):
        if x % 70 < 48:
            for y in range(8, 25):
                pixels[x, y] = (0, 0, 0)
    polygon = LinePolygon([(0, 0), (700, 0), (700, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", image, polygon)
    predictor = ConsensusPredictor()
    recognizer = VietOcrRecognizer(
        Settings(split_wide_crops=False, hallucination_guard_enabled=True),
        predictor=predictor,
    )

    result = recognizer.recognize([crop])

    assert result[0].text == "trao đổi, cung cấp"
    assert result[0].error_code == "unsupported_ocr_insertion_removed"
    assert predictor.predict_calls == 1
    assert recognizer.last_batch_metrics["hallucination_guard_consensus_count"] == 1
    assert recognizer.last_batch_metrics["hallucination_guard_removed_token_count"] == 1
    event = recognizer.last_page_metrics[0]["hallucination_guard_events"][0]
    assert event["removed_tokens"] == ["xyzzy"]


def test_morphology_aware_split_prefers_real_interword_gap():
    from government_ocr_text_api.vietocr_recognizer import _split_wide_image

    image = Image.new("RGB", (900, 32), "white")
    pixels = image.load()
    # Hai khối chữ lớn, khe giữa đủ rộng. Các khe 1px bên trong khối mô phỏng
    # nét ký tự không được chọn làm seam.
    for start, end in [(10, 420), (480, 890)]:
        for x in range(start, end):
            if x % 17 == 0:
                continue
            for y in range(7, 26):
                pixels[x, y] = (0, 0, 0)

    segments = _split_wide_image(
        image,
        image_height=32,
        image_max_width=512,
        target_ratio=0.92,
        max_segments=6,
        valley_max_ink_ratio=0.02,
        overlap_height_ratio=0.0,
    )

    assert len(segments) >= 2
    first_cut = segments[0].core_right
    assert 420 <= first_cut <= 480



def test_decoder_evidence_trims_low_support_suffix_after_visual_end():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "Nội dung đúng xyz"
    prefix = len("Nội dung đúng ")
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.95] * prefix + [0.30, 0.28, 0.25]),
        attention_centers=tuple(
            [i / max(1, prefix) * 0.82 for i in range(prefix)] + [0.94, 0.945, 0.947]
        ),
        attention_spreads=tuple([0.05] * len(text)),
        ink_support=tuple([0.75] * prefix + [0.02, 0.02, 0.01]),
        last_ink_position=0.91,
    )

    decision = _decoder_evidence_decision(trace, Settings())

    assert decision is not None
    assert decision.text == "Nội dung đúng"
    assert decision.removed_text == "xyz"
    assert decision.reason in {"visual_evidence_exhausted", "attention_stall"}


def test_decoder_evidence_keeps_numeric_suffix_in_warning_only_mode():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "Hiệu lực 2022.000"
    prefix = len("Hiệu lực 2022.")
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.95] * prefix + [0.20, 0.20, 0.20]),
        attention_centers=tuple([0.5] * prefix + [0.96, 0.96, 0.96]),
        attention_spreads=tuple([0.05] * len(text)),
        ink_support=tuple([0.7] * prefix + [0.01, 0.01, 0.01]),
        last_ink_position=0.92,
    )

    decision = _decoder_evidence_decision(
        trace,
        Settings(decoder_evidence_numeric_apply_changes=True),
    )

    assert decision is not None
    assert decision.numeric_only_warning is True
    assert decision.removed_text == "000"


def test_near_repetitive_suffix_is_detected_without_word_blacklist():
    from government_ocr_text_api.vietocr_recognizer import _near_repetitive_suffix_start

    text = (
        "Nội dung hợp lệ trang bố chiến bang bố chiến biển bang bố chiến "
        "thung bố chiến binh thung bố chiến binh"
    )
    start = _near_repetitive_suffix_start(text, 0.74, 8)

    assert start is not None
    assert text[start:].startswith("trang")


def _install_fake_vietocr_process_input(monkeypatch):
    import sys
    from types import ModuleType

    import torch

    translate_module = ModuleType("vietocr.tool.translate")
    translate_module.process_input = lambda image, *args: torch.zeros((1, 3, 32, 100))
    vietocr_module = ModuleType("vietocr")
    vietocr_module.__path__ = []
    tool_module = ModuleType("vietocr.tool")
    tool_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vietocr", vietocr_module)
    monkeypatch.setitem(sys.modules, "vietocr.tool", tool_module)
    monkeypatch.setitem(sys.modules, "vietocr.tool.translate", translate_module)


def _contract_faithful_trace_predictor(
    monkeypatch,
    *,
    token_ids,
    token_text,
    attention_3d=False,
    malformed_decoder=False,
):
    from types import SimpleNamespace

    import torch

    _install_fake_vietocr_process_input(monkeypatch)
    i2c = {index + 4: char for index, char in enumerate(token_text)}
    seen_hidden_steps = []

    class Decoder:
        def __call__(self, input_token, hidden, encoder_outputs):
            assert input_token.shape == (1,)
            assert hidden.shape == (1, 4)
            assert encoder_outputs.shape == (10, 1, 4)
            step = int(hidden[0, 0].item())
            seen_hidden_steps.append(step)
            token = token_ids[step] if step < len(token_ids) else 2
            prediction = torch.full((1, 32), -20.0)
            prediction[0, token] = 8.0 if step < 6 else -0.2
            if step >= 6:
                prediction[0, 4] = -0.4
            attention = torch.zeros((1, 10))
            attention[0, min(step + 1, 9)] = 1.0
            if step >= 6:
                attention.zero_()
                attention[0, 9] = 1.0
            if attention_3d:
                attention = attention.unsqueeze(1)
            next_hidden = hidden.clone()
            next_hidden[0, 0] += 1
            if malformed_decoder:
                return prediction, attention
            return prediction, next_hidden, attention

    transformer = SimpleNamespace(
        decoder=Decoder(),
        forward_encoder=lambda src: (torch.zeros((1, 4)), torch.zeros((10, 1, 4))),
    )
    model = SimpleNamespace(
        transformer=transformer,
        cnn=lambda tensor: torch.zeros((10, 1, 4)),
        eval=lambda: None,
    )
    predictor = SimpleNamespace(
        model=model,
        vocab=SimpleNamespace(i2c=i2c),
        config={
            "device": "cpu",
            "dataset": {
                "image_height": 32,
                "image_min_width": 32,
                "image_max_width": 512,
            },
        },
    )
    return predictor, seen_hidden_steps


def _ink_image(width=100):
    image = Image.new("RGB", (width, 32), "white")
    pixels = image.load()
    for x in range(5, int(width * 0.65)):
        for y in range(8, 25):
            pixels[x, y] = (0, 0, 0)
    return image


def test_seq2seq_trace_uses_single_token_and_recurrent_hidden_contract(monkeypatch):
    from government_ocr_text_api.vietocr_recognizer import (
        _decoder_evidence_decision,
        _trace_seq2seq_attention_detailed,
    )

    token_text = "abcdefxyz"
    token_ids = [index + 4 for index in range(len(token_text))]
    predictor, seen_hidden_steps = _contract_faithful_trace_predictor(
        monkeypatch,
        token_ids=token_ids,
        token_text=token_text,
    )
    settings = Settings(
        decoder_evidence_min_prefix_chars=4,
        decoder_evidence_window_tokens=3,
        decoder_evidence_min_unsupported_tokens=3,
        decoder_evidence_low_token_probability=0.60,
        decoder_evidence_low_ink_support=0.10,
    )

    outcome = _trace_seq2seq_attention_detailed(predictor, _ink_image(), settings)

    assert outcome.error_type is None
    assert outcome.trace is not None
    assert outcome.trace.raw_text == "abcdefxyz"
    assert seen_hidden_steps == list(range(10))  # 9 ký tự + bước EOS
    decision = _decoder_evidence_decision(outcome.trace, settings)
    assert decision is not None
    assert decision.text == "abcdef"
    assert decision.removed_text == "xyz"


def test_seq2seq_trace_accepts_singleton_3d_attention(monkeypatch):
    from government_ocr_text_api.vietocr_recognizer import (
        _trace_seq2seq_attention_detailed,
    )

    predictor, _ = _contract_faithful_trace_predictor(
        monkeypatch,
        token_ids=[4, 5],
        token_text="ab",
        attention_3d=True,
    )

    outcome = _trace_seq2seq_attention_detailed(predictor, _ink_image(), Settings())

    assert outcome.trace is not None
    assert outcome.trace.raw_text == "ab"
    assert len(outcome.trace.attention_centers) == 2


def test_seq2seq_trace_stops_at_eos_and_does_not_emit_special_tokens(monkeypatch):
    from government_ocr_text_api.vietocr_recognizer import (
        _trace_seq2seq_attention_detailed,
    )

    # Token 3 là special token và không được đưa vào raw_text; token 2 là EOS.
    predictor, seen_steps = _contract_faithful_trace_predictor(
        monkeypatch,
        token_ids=[3, 4, 2, 5],
        token_text="ab",
    )

    outcome = _trace_seq2seq_attention_detailed(predictor, _ink_image(), Settings())

    assert outcome.trace is not None
    assert outcome.trace.raw_text == "a"
    assert len(outcome.trace.token_probabilities) == 1
    assert seen_steps == [0, 1, 2]


def test_seq2seq_trace_reports_decoder_contract_error_instead_of_silently_failing(
    monkeypatch,
):
    from government_ocr_text_api.vietocr_recognizer import (
        _trace_seq2seq_attention_detailed,
    )

    predictor, _ = _contract_faithful_trace_predictor(
        monkeypatch,
        token_ids=[4],
        token_text="a",
        malformed_decoder=True,
    )

    outcome = _trace_seq2seq_attention_detailed(predictor, _ink_image(), Settings())

    assert outcome.trace is None
    assert outcome.fatal is True
    assert outcome.error_type == "_DecoderContractError"
    assert "must return" in outcome.error_message


def test_decoder_evidence_circuit_breaker_stops_repeating_fatal_adapter_errors():
    class LowConfidencePredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            self.batch_sizes.append(len(images))
            return ["nội dung nghi ngờ"] * len(images), [0.20] * len(images)

    polygon = LinePolygon([(0, 0), (1200, 0), (1200, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(f"p{page:04d}-l0000", _synthetic_text_line(1200), polygon)
        for page in range(2)
    ]
    recognizer = VietOcrRecognizer(
        Settings(
            decoder_evidence_enabled=True,
            decoder_evidence_max_checks_per_page=6,
            hallucination_guard_enabled=False,
        ),
        predictor=LowConfidencePredictor(),
    )

    recognizer.recognize(crops)

    metrics = recognizer.last_batch_metrics
    assert metrics["decoder_evidence_candidate_count"] > 1
    assert metrics["decoder_evidence_trace_count"] == 1
    assert metrics["decoder_evidence_supported_count"] == 0
    assert metrics["decoder_evidence_circuit_breaker_count"] == 1
    assert metrics["decoder_evidence_disabled_reason"] == "predictor_missing_model_vocab_or_config"


def test_trace_text_mismatch_never_mutates_main_greedy_output(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    class LowConfidencePredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            self.batch_sizes.append(len(images))
            return ["nội dung gốc xyz"] * len(images), [0.20] * len(images)

    trace = module._DecoderEvidenceTrace(
        raw_text="nội dung khác xyz",
        token_probabilities=tuple([0.2] * len("nội dung khác xyz")),
        attention_centers=tuple([0.95] * len("nội dung khác xyz")),
        attention_spreads=tuple([0.01] * len("nội dung khác xyz")),
        ink_support=tuple([0.01] * len("nội dung khác xyz")),
        last_ink_position=0.90,
    )
    monkeypatch.setattr(
        module,
        "_trace_seq2seq_attention_batch_detailed",
        lambda predictor, images, settings: [
            module._DecoderEvidenceTraceOutcome(trace=trace) for _ in images
        ],
    )
    polygon = LinePolygon([(0, 0), (1200, 0), (1200, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", _synthetic_text_line(1200), polygon)
    recognizer = VietOcrRecognizer(
        Settings(
            decoder_evidence_enabled=True,
            decoder_evidence_full_coverage=False,
            decoder_evidence_max_checks_per_page=1,
            hallucination_guard_enabled=False,
        ),
        predictor=LowConfidencePredictor(),
    )

    result = recognizer.recognize([crop])

    assert "nội dung gốc" in result[0].text
    assert "nội dung khác" not in result[0].text
    # Selective mode traces the top-risk segment plus all segments of the same
    # split line so cross-segment anchors remain available. Every mismatched
    # trace must still be fail-closed and preserve greedy text.
    assert recognizer.last_batch_metrics["decoder_evidence_seed_selected_count"] == 1
    assert recognizer.last_batch_metrics["decoder_evidence_context_forced_count"] >= 1
    assert recognizer.last_batch_metrics["decoder_evidence_trace_mismatch_count"] >= 1
    assert recognizer.last_batch_metrics["decoder_evidence_trimmed_count"] == 0


def test_decoder_evidence_does_not_trim_when_suffix_probability_is_high():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "Nội dung đúng xyz"
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.95] * len(text)),
        attention_centers=tuple([0.96] * len(text)),
        attention_spreads=tuple([0.01] * len(text)),
        ink_support=tuple([0.01] * len(text)),
        last_ink_position=0.90,
    )

    assert _decoder_evidence_decision(trace, Settings()) is None


def test_decoder_evidence_does_not_trim_when_suffix_has_strong_ink_support():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "Nội dung đúng xyz"
    prefix = len("Nội dung đúng ")
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.95] * prefix + [0.2, 0.2, 0.2]),
        attention_centers=tuple([0.5] * prefix + [0.96, 0.96, 0.96]),
        attention_spreads=tuple([0.01] * len(text)),
        ink_support=tuple([0.7] * len(text)),
        last_ink_position=0.90,
    )

    assert _decoder_evidence_decision(trace, Settings()) is None


def test_near_loop_text_alone_is_not_enough_without_visual_evidence():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = (
        "Nội dung hợp lệ trang bố chiến bang bố chiến biển bang bố chiến "
        "thung bố chiến binh thung bố chiến binh"
    )
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.95] * len(text)),
        attention_centers=tuple([index / len(text) for index in range(len(text))]),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple([0.8] * len(text)),
        last_ink_position=0.98,
    )

    assert _decoder_evidence_decision(trace, Settings()) is None


def test_legal_parallel_phrases_are_not_misclassified_as_near_loop():
    from government_ocr_text_api.vietocr_recognizer import _near_repetitive_suffix_start

    text = (
        "theo pháp luật về bảo vệ an ninh quốc gia, phòng, chống khủng bố; "
        "phòng, chống rửa tiền"
    )

    assert _near_repetitive_suffix_start(text, 0.74, 8) is None


def test_installed_vietocr_seq2seq_decoder_contract_without_checkpoint():
    import pytest
    import torch

    module = pytest.importorskip("vietocr.model.seqmodel.seq2seq")
    seq2seq = module.Seq2Seq(
        vocab_size=20,
        encoder_hidden=4,
        decoder_hidden=4,
        img_channel=8,
        decoder_embedded=4,
        dropout=0.0,
    )
    source = torch.randn((7, 1, 8))
    hidden, encoder_outputs = seq2seq.forward_encoder(source)

    prediction, next_hidden, attention = seq2seq.decoder(
        torch.tensor([1]), hidden, encoder_outputs
    )

    assert prediction.shape == (1, 20)
    assert next_hidden.shape == hidden.shape == (1, 4)
    assert attention.shape == (1, 7)


def test_decoder_evidence_expands_partial_suffix_to_full_attention_cluster_word():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "quản lý kịp thời trao đổi, người thuy"
    word_start = text.index("người")
    tail_start = text.index("thuy")
    probabilities = [0.95] * len(text)
    supports = [0.72] * len(text)
    centers = [min(0.82, 0.10 + index * 0.018) for index in range(len(text))]
    for index in range(word_start, tail_start):
        probabilities[index] = 0.74
        supports[index] = 0.035
        centers[index] = 0.945
    for index in range(tail_start, len(text)):
        probabilities[index] = 0.25
        supports[index] = 0.02
        centers[index] = 0.95
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple(probabilities),
        attention_centers=tuple(centers),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple(supports),
        last_ink_position=0.90,
    )

    decision = _decoder_evidence_decision(trace, Settings())

    assert decision is not None
    assert decision.text == "quản lý kịp thời trao đổi,"
    assert decision.removed_text == "người thuy"
    assert decision.expanded_word_count == 1


def test_decoder_evidence_does_not_expand_into_visually_supported_previous_word():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "theo quy định của pháp luật chiến"
    tail_start = text.index("chiến")
    probabilities = [0.95] * tail_start + [0.22] * (len(text) - tail_start)
    supports = [0.70] * tail_start + [0.02] * (len(text) - tail_start)
    centers = [index / max(1, len(text)) * 0.82 for index in range(tail_start)]
    centers += [0.95] * (len(text) - tail_start)
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple(probabilities),
        attention_centers=tuple(centers),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple(supports),
        last_ink_position=0.90,
    )

    decision = _decoder_evidence_decision(trace, Settings())

    assert decision is not None
    assert decision.text == "theo quy định của pháp luật"
    assert decision.removed_text == "chiến"
    assert decision.expanded_word_count == 0


def test_relative_threshold_detects_tail_above_absolute_probability_limit():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "Nội dung rõ xyz"
    prefix = len("Nội dung rõ ")
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.96] * prefix + [0.61, 0.60, 0.59]),
        attention_centers=tuple([0.20 + index * 0.04 for index in range(prefix)] + [0.95] * 3),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple([0.75] * prefix + [0.06, 0.05, 0.05]),
        last_ink_position=0.91,
    )

    decision = _decoder_evidence_decision(trace, Settings())

    assert decision is not None
    assert decision.text == "Nội dung rõ"
    assert decision.probability_threshold > 0.61


def test_relative_threshold_does_not_trim_uniformly_weak_line_without_stable_prefix():
    from government_ocr_text_api.vietocr_recognizer import (
        _DecoderEvidenceTrace,
        _decoder_evidence_decision,
    )

    text = "toàn bộ dòng đều mờ"
    trace = _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.30] * len(text)),
        attention_centers=tuple([0.95] * len(text)),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple([0.02] * len(text)),
        last_ink_position=0.90,
    )

    assert _decoder_evidence_decision(trace, Settings()) is None


def test_batch_trace_decodes_multiple_same_size_images_in_one_decoder_loop(monkeypatch):
    import torch
    from types import SimpleNamespace

    from government_ocr_text_api.vietocr_recognizer import (
        _trace_seq2seq_attention_batch_detailed,
    )

    _install_fake_vietocr_process_input(monkeypatch)
    calls = []

    class Decoder:
        def __call__(self, input_token, hidden, encoder_outputs):
            batch = input_token.shape[0]
            step = int(hidden[0, 0].item())
            calls.append(batch)
            token = [4, 5, 2][min(step, 2)]
            prediction = torch.full((batch, 16), -20.0)
            prediction[:, token] = 8.0
            attention = torch.zeros((batch, 10))
            attention[:, min(step + 2, 9)] = 1.0
            next_hidden = hidden.clone()
            next_hidden[:, 0] += 1
            return prediction, next_hidden, attention

    transformer = SimpleNamespace(
        decoder=Decoder(),
        forward_encoder=lambda src: (
            torch.zeros((src.shape[1], 4)),
            torch.zeros((10, src.shape[1], 4)),
        ),
    )
    model = SimpleNamespace(
        transformer=transformer,
        cnn=lambda tensor: torch.zeros((10, tensor.shape[0], 4)),
        eval=lambda: None,
    )
    predictor = SimpleNamespace(
        model=model,
        vocab=SimpleNamespace(i2c={4: "a", 5: "b"}),
        config={
            "device": "cpu",
            "dataset": {
                "image_height": 32,
                "image_min_width": 32,
                "image_max_width": 512,
            },
        },
    )

    outcomes = _trace_seq2seq_attention_batch_detailed(
        predictor, [_ink_image(), _ink_image(), _ink_image()], Settings()
    )

    assert [outcome.trace.raw_text for outcome in outcomes] == ["ab", "ab", "ab"]
    assert calls == [3, 3, 3]


def test_recognizer_batches_decoder_evidence_candidates_and_reports_coverage(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    class LowConfidencePredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            self.batch_sizes.append(len(images))
            return ["nội dung nghi ngờ"] * len(images), [0.20] * len(images)

    calls = []

    def fake_batch_trace(predictor, images, settings):
        calls.append(len(images))
        trace = module._DecoderEvidenceTrace(
            raw_text="nội dung nghi ngờ",
            token_probabilities=tuple([0.95] * len("nội dung nghi ngờ")),
            attention_centers=tuple([0.5] * len("nội dung nghi ngờ")),
            attention_spreads=tuple([0.03] * len("nội dung nghi ngờ")),
            ink_support=tuple([0.8] * len("nội dung nghi ngờ")),
            last_ink_position=0.9,
        )
        return [module._DecoderEvidenceTraceOutcome(trace=trace) for _ in images]

    monkeypatch.setattr(module, "_trace_seq2seq_attention_batch_detailed", fake_batch_trace)
    polygon = LinePolygon([(0, 0), (300, 0), (300, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(f"p0000-l{index:04d}", Image.new("RGB", (300, 32), "white"), polygon)
        for index in range(8)
    ]
    recognizer = VietOcrRecognizer(
        Settings(
            split_wide_crops=False,
            decoder_evidence_max_checks_per_page=12,
            decoder_evidence_trace_batch_size=12,
            hallucination_guard_enabled=False,
        ),
        predictor=LowConfidencePredictor(),
    )

    recognizer.recognize(crops)

    metrics = recognizer.last_batch_metrics
    assert calls == [8]
    assert metrics["decoder_evidence_candidate_count"] == 8
    assert metrics["decoder_evidence_selected_count"] == 8
    assert metrics["decoder_evidence_unchecked_candidate_count"] == 0
    assert metrics["decoder_evidence_trace_count"] == 8
    assert metrics["decoder_evidence_trace_batch_count"] == 1
    assert metrics["decoder_evidence_trace_batch_size_max"] == 8


def test_1356_defaults_use_selective_evidence_with_split_line_context():
    settings = Settings()
    assert settings.hallucination_guard_enabled is False
    assert settings.decoder_evidence_full_coverage is False
    assert settings.decoder_evidence_max_checks_per_page == 8
    assert settings.decoder_evidence_include_split_line_context is True
    assert settings.decoder_evidence_trace_batch_size == 12
    assert settings.decoder_evidence_cluster_word_expansion_enabled is True
    assert settings.decoder_evidence_midline_enabled is True


def test_numeric_warning_event_distinguishes_proposal_from_applied_text(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    class NumericPredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            self.batch_sizes.append(len(images))
            return ["thoại, fax.000"] * len(images), [0.20] * len(images)

    text = "thoại, fax.000"
    prefix = len("thoại, fax.")
    trace = module._DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple([0.95] * prefix + [0.20, 0.20, 0.20]),
        attention_centers=tuple([0.5] * prefix + [0.96, 0.96, 0.96]),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple([0.7] * prefix + [0.01, 0.01, 0.01]),
        last_ink_position=0.92,
    )
    monkeypatch.setattr(
        module,
        "_trace_seq2seq_attention_batch_detailed",
        lambda predictor, images, settings: [
            module._DecoderEvidenceTraceOutcome(trace=trace) for _ in images
        ],
    )
    polygon = LinePolygon([(0, 0), (300, 0), (300, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", Image.new("RGB", (300, 32), "white"), polygon)
    recognizer = VietOcrRecognizer(
        Settings(
            split_wide_crops=False,
            hallucination_guard_enabled=False,
            decoder_evidence_max_checks_per_page=1,
        ),
        predictor=NumericPredictor(),
    )

    result = recognizer.recognize([crop])
    event = recognizer.last_page_metrics[0]["decoder_evidence_events"][0]

    assert result[0].text == text
    assert event["proposed_text"] == "thoại, fax."
    assert event["validated_text"] == text
    assert event["applied"] is False
    assert event["numeric_warning_only"] is True


def test_installed_vietocr_seq2seq_batch_decoder_contract_without_checkpoint():
    import pytest
    import torch

    module = pytest.importorskip("vietocr.model.seqmodel.seq2seq")
    seq2seq = module.Seq2Seq(
        vocab_size=20,
        encoder_hidden=4,
        decoder_hidden=4,
        img_channel=8,
        decoder_embedded=4,
        dropout=0.0,
    )
    source = torch.randn((7, 3, 8))
    hidden, encoder_outputs = seq2seq.forward_encoder(source)

    prediction, next_hidden, attention = seq2seq.decoder(
        torch.tensor([1, 1, 1]), hidden, encoder_outputs
    )

    assert prediction.shape == (3, 20)
    assert next_hidden.shape == hidden.shape == (3, 4)
    assert attention.shape == (3, 7)


def test_batch_trace_handles_heterogeneous_eos_without_cross_sample_text(monkeypatch):
    import torch
    from types import SimpleNamespace

    from government_ocr_text_api.vietocr_recognizer import (
        _trace_seq2seq_attention_batch_detailed,
    )

    _install_fake_vietocr_process_input(monkeypatch)

    class Decoder:
        def __call__(self, input_token, hidden, encoder_outputs):
            batch = input_token.shape[0]
            steps = hidden[:, 0].to(dtype=torch.long)
            prediction = torch.full((batch, 16), -20.0)
            # sample 0: a, EOS; sample 1: a, b, EOS
            sequences = ([4, 2, 2], [4, 5, 2])
            for index in range(batch):
                token = sequences[index][min(int(steps[index].item()), 2)]
                prediction[index, token] = 8.0
            attention = torch.zeros((batch, 10))
            attention[:, 5] = 1.0
            next_hidden = hidden.clone()
            next_hidden[:, 0] += 1
            return prediction, next_hidden, attention

    transformer = SimpleNamespace(
        decoder=Decoder(),
        forward_encoder=lambda src: (
            torch.zeros((src.shape[1], 4)),
            torch.zeros((10, src.shape[1], 4)),
        ),
    )
    predictor = SimpleNamespace(
        model=SimpleNamespace(
            transformer=transformer,
            cnn=lambda tensor: torch.zeros((10, tensor.shape[0], 4)),
            eval=lambda: None,
        ),
        vocab=SimpleNamespace(i2c={4: "a", 5: "b"}),
        config={
            "device": "cpu",
            "dataset": {
                "image_height": 32,
                "image_min_width": 32,
                "image_max_width": 512,
            },
        },
    )

    outcomes = _trace_seq2seq_attention_batch_detailed(
        predictor, [_ink_image(), _ink_image()], Settings()
    )

    assert [outcome.trace.raw_text for outcome in outcomes] == ["a", "ab"]



def _trace_with_word_evidence(
    words,
    weak_word_indexes=(),
    *,
    weak_center=0.31,
    weak_probability=0.20,
    weak_support=0.02,
    supported_probability=0.95,
    supported_support=0.72,
):
    from government_ocr_text_api.vietocr_recognizer import _DecoderEvidenceTrace

    text = " ".join(words)
    probabilities = [supported_probability] * len(text)
    supports = [supported_support] * len(text)
    centers = [0.0] * len(text)
    matches = list(__import__("re").finditer(r"\w+", text, __import__("re").UNICODE))
    supported_centers = [0.12 + index * 0.18 for index in range(len(matches))]
    for index, match in enumerate(matches):
        center = weak_center if index in weak_word_indexes else supported_centers[index]
        for char_index in range(match.start(), match.end()):
            centers[char_index] = center
            if index in weak_word_indexes:
                probabilities[char_index] = weak_probability
                supports[char_index] = weak_support
    # Spaces carry the nearest previous attention position but are not used as words.
    last = supported_centers[0]
    for index, value in enumerate(centers):
        if value:
            last = value
        else:
            centers[index] = last
    return _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple(probabilities),
        attention_centers=tuple(centers),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple(supports),
        last_ink_position=0.92,
    )


def test_midline_guard_removes_generic_unsupported_span_between_supported_anchors():
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    trace = _trace_with_word_evidence(
        ["alpha", "bravo", "charlie", "delta", "echo"],
        weak_word_indexes={2},
        weak_center=0.30,
    )

    decision = _decoder_evidence_decision(trace, Settings())

    assert decision is not None
    assert decision.span_kind == "midline"
    assert decision.reason == "unsupported_midline_span"
    assert decision.removed_text == "charlie"
    assert decision.text == "alpha bravo delta echo"
    assert decision.left_anchor_ink_support_mean > decision.span_ink_support_mean
    assert decision.right_anchor_ink_support_mean > decision.span_ink_support_mean


def test_midline_guard_is_lexically_invariant_for_equal_evidence_patterns():
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    cases = [
        (["alpha", "bravo", "charlie", "delta", "echo"], "charlie"),
        (["satin", "cobalt", "marbles", "silver", "ember"], "marbles"),
        (["mango", "violet", "orchard", "copper", "flame"], "orchard"),
    ]
    decisions = []
    for words, removed in cases:
        decision = _decoder_evidence_decision(
            _trace_with_word_evidence(words, weak_word_indexes={2}, weak_center=0.30),
            Settings(),
        )
        assert decision is not None
        assert decision.removed_text == removed
        assert decision.span_kind == "midline"
        decisions.append(decision)

    assert {decision.reason for decision in decisions} == {"unsupported_midline_span"}
    assert len({decision.span_ink_support_mean for decision in decisions}) == 1


def test_midline_guard_keeps_visually_supported_middle_word():
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    trace = _trace_with_word_evidence(
        ["alpha", "bravo", "charlie", "delta", "echo"],
        weak_word_indexes=set(),
    )

    assert _decoder_evidence_decision(trace, Settings()) is None


def test_midline_guard_requires_supported_right_recovery():
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    trace = _trace_with_word_evidence(
        ["alpha", "bravo", "charlie", "delta", "echo"],
        weak_word_indexes={2, 3, 4},
        weak_center=0.30,
    )

    decision = _decoder_evidence_decision(trace, Settings())
    assert decision is None or decision.span_kind != "midline"


def test_midline_numeric_span_remains_warning_only():
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    trace = _trace_with_word_evidence(
        ["alpha", "bravo", "12345", "delta", "echo"],
        weak_word_indexes={2},
        weak_center=0.30,
    )

    decision = _decoder_evidence_decision(
        trace,
        Settings(decoder_evidence_numeric_apply_changes=True),
    )

    assert decision is not None
    assert decision.span_kind == "midline"
    assert decision.numeric_only_warning is True


def test_semantic_runtime_has_no_corpus_specific_replacement_literals():
    from pathlib import Path
    import government_ocr_text_api.vietocr_recognizer as module

    source = Path(module.__file__).read_text(encoding="utf-8").casefold()
    corpus_artifacts = (
        "người thuy",
        "cung nghiệt",
        "pháp luật chiến",
        "nhanh thuyên thuy nghiệp",
        "001000001194",
        "fax.000",
    )
    assert not [artifact for artifact in corpus_artifacts if artifact in source]


def test_full_coverage_traces_high_confidence_segments_when_limit_is_unbounded(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    class HighConfidencePredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            self.batch_sizes.append(len(images))
            return ["generic supported content"] * len(images), [0.99] * len(images)

    calls = []

    def fake_batch_trace(predictor, images, settings):
        calls.append(len(images))
        text = "generic supported content"
        trace = module._DecoderEvidenceTrace(
            raw_text=text,
            token_probabilities=tuple([0.99] * len(text)),
            attention_centers=tuple(
                [0.10 + index * 0.70 / max(1, len(text) - 1) for index in range(len(text))]
            ),
            attention_spreads=tuple([0.03] * len(text)),
            ink_support=tuple([0.8] * len(text)),
            last_ink_position=0.9,
        )
        return [module._DecoderEvidenceTraceOutcome(trace=trace) for _ in images]

    monkeypatch.setattr(module, "_trace_seq2seq_attention_batch_detailed", fake_batch_trace)
    polygon = LinePolygon([(0, 0), (300, 0), (300, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(f"p0000-l{index:04d}", Image.new("RGB", (300, 32), "white"), polygon)
        for index in range(5)
    ]
    recognizer = VietOcrRecognizer(
        Settings(
            split_wide_crops=False,
            decoder_evidence_full_coverage=True,
            decoder_evidence_max_checks_per_page=0,
            hallucination_guard_enabled=False,
        ),
        predictor=HighConfidencePredictor(),
    )

    recognizer.recognize(crops)

    metrics = recognizer.last_batch_metrics
    assert calls == [5]
    assert metrics["decoder_evidence_candidate_count"] == 5
    assert metrics["decoder_evidence_selected_count"] == 5
    assert metrics["decoder_evidence_unchecked_candidate_count"] == 0



def test_midline_guard_removes_multiword_span_without_phrase_fixture():
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    words = ["anchor", "before", "lumen", "mirth", "sable", "after", "stable"]
    trace = _trace_with_word_evidence(
        words,
        weak_word_indexes={2, 3, 4},
        weak_center=0.31,
    )

    decision = _decoder_evidence_decision(trace, Settings())

    assert decision is not None
    assert decision.span_kind == "midline"
    assert decision.removed_text == "lumen mirth sable"
    assert decision.text == "anchor before after stable"


def test_midline_guard_generalizes_across_random_lexical_content():
    import random
    import string
    from government_ocr_text_api.vietocr_recognizer import _decoder_evidence_decision

    randomizer = random.Random(1353)

    def token(length):
        return "".join(randomizer.choice(string.ascii_lowercase) for _ in range(length))

    for _ in range(25):
        words = [token(6), token(7), token(8), token(6), token(7)]
        trace = _trace_with_word_evidence(
            words,
            weak_word_indexes={2},
            weak_center=0.30,
        )
        decision = _decoder_evidence_decision(trace, Settings())
        assert decision is not None
        assert decision.removed_text == words[2]
        assert decision.text == " ".join((words[0], words[1], words[3], words[4]))


def test_full_coverage_overrides_legacy_per_page_limit(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    class PredictorWithStableText(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            return ["stable generic content"] * len(images), [0.99] * len(images)

    calls = []

    def fake_batch_trace(predictor, images, settings):
        calls.append(len(images))
        text = "stable generic content"
        trace = module._DecoderEvidenceTrace(
            raw_text=text,
            token_probabilities=tuple([0.99] * len(text)),
            attention_centers=tuple([0.2 + 0.5 * i / len(text) for i in range(len(text))]),
            attention_spreads=tuple([0.03] * len(text)),
            ink_support=tuple([0.8] * len(text)),
            last_ink_position=0.9,
        )
        return [module._DecoderEvidenceTraceOutcome(trace=trace) for _ in images]

    monkeypatch.setattr(module, "_trace_seq2seq_attention_batch_detailed", fake_batch_trace)
    polygon = LinePolygon([(0, 0), (300, 0), (300, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(f"p0000-l{index:04d}", Image.new("RGB", (300, 32), "white"), polygon)
        for index in range(6)
    ]
    recognizer = VietOcrRecognizer(
        Settings(
            split_wide_crops=False,
            decoder_evidence_full_coverage=True,
            decoder_evidence_max_checks_per_page=1,
        ),
        predictor=PredictorWithStableText(),
    )

    recognizer.recognize(crops)

    assert sum(calls) == 6
    assert recognizer.last_batch_metrics["decoder_evidence_selected_count"] == 6
    assert recognizer.last_batch_metrics["decoder_evidence_unchecked_candidate_count"] == 0


def _grounded_trace(text, weak_words=(), weak_center=0.72):
    import re
    from government_ocr_text_api.vietocr_recognizer import _DecoderEvidenceTrace

    probabilities = [0.94] * len(text)
    supports = [0.72] * len(text)
    centers = [0.10] * len(text)
    coverage = [0.66] * len(text)
    reuse = [0.10] * len(text)
    words = list(re.finditer(r"\w+", text, re.UNICODE))
    for word_index, match in enumerate(words):
        center = 0.12 + word_index * 0.16
        is_weak = word_index in set(weak_words)
        for char_index in range(match.start(), match.end()):
            centers[char_index] = weak_center if is_weak else center
            if is_weak:
                probabilities[char_index] = 0.36
                supports[char_index] = 0.025
                coverage[char_index] = 0.035
                reuse[char_index] = 0.88
    last = centers[0] if centers else 0.0
    for index, value in enumerate(centers):
        if text[index].isspace():
            centers[index] = last
        else:
            last = value
    return _DecoderEvidenceTrace(
        raw_text=text,
        token_probabilities=tuple(probabilities),
        attention_centers=tuple(centers),
        attention_spreads=tuple([0.03] * len(text)),
        ink_support=tuple(supports),
        last_ink_position=0.93,
        novel_ink_support=tuple(coverage),
        visual_coverage_gain=tuple(coverage),
        reused_attention_ratio=tuple(reuse),
    )


def _segment_for_cross(original_index, segment_index, left, right):
    from government_ocr_text_api.vietocr_recognizer import _Segment

    return _Segment(
        original_index=original_index,
        segment_index=segment_index,
        page_index=0,
        image=Image.new("RGB", (right - left, 32), "white"),
        resized_width=right - left,
        was_split=True,
        source_left=left,
        source_right=right,
        left_overlap=0,
        right_overlap=0,
        ink_ratio=0.08,
        ink_columns_ratio=0.8,
        is_tail=segment_index == 1,
    )


def test_visual_grounding_step_distinguishes_new_from_reused_attention():
    import numpy as np
    from government_ocr_text_api.vietocr_recognizer import _visual_grounding_step

    ink = np.ones(8, dtype=np.float32)
    previous = np.zeros(8, dtype=np.float32)
    first = np.zeros(8, dtype=np.float32)
    first[1:3] = 0.5
    novel, gain, reuse = _visual_grounding_step(first, ink, previous)
    assert novel > 0.95
    assert gain > 0.9
    assert reuse == 0.0

    previous = np.maximum(previous, first)
    repeated = first.copy()
    novel2, gain2, reuse2 = _visual_grounding_step(repeated, ink, previous)
    assert novel2 < 0.05
    assert gain2 < 0.01
    assert reuse2 > 0.95


def test_global_attention_mapping_uses_original_line_coordinate():
    import numpy as np
    from government_ocr_text_api.vietocr_recognizer import _global_attention_centers

    trace = _grounded_trace("alpha bravo")
    segment = _segment_for_cross(0, 1, 500, 1000)
    mapped = _global_attention_centers(trace, segment, 1000, 500)
    assert mapped.size == len(trace.raw_text)
    assert float(mapped.min()) >= 0.5
    assert float(mapped.max()) <= 1.0
    assert np.all(np.diff(mapped[np.isfinite(mapped)]) >= -0.25)


def test_cross_segment_guard_removes_multiword_weak_tail_with_right_anchor():
    from government_ocr_text_api.vietocr_recognizer import _cross_segment_suffix_decision

    left = _grounded_trace("alpha bravo lumen sable", weak_words={2, 3}, weak_center=0.72)
    right = _grounded_trace("charlie delta", weak_words=set())
    decision = _cross_segment_suffix_decision(
        left,
        right,
        _segment_for_cross(0, 0, 0, 500),
        _segment_for_cross(0, 1, 500, 1000),
        1000,
        500,
        500,
        Settings(),
    )
    assert decision is not None
    assert decision.span_kind == "cross_segment"
    assert decision.reason == "cross_segment_visual_gap"
    assert decision.removed_text == "lumen sable"
    assert decision.text == "alpha bravo"
    assert decision.span_visual_coverage_gain_mean < decision.left_anchor_visual_coverage_gain_mean
    assert decision.span_reused_attention_ratio_mean > 0.6


def test_cross_segment_guard_is_lexically_invariant():
    from government_ocr_text_api.vietocr_recognizer import _cross_segment_suffix_decision

    cases = [
        ("alpha bravo lumen sable", "charlie delta", "lumen sable"),
        ("cobalt ember satin mirth", "violet copper", "satin mirth"),
        ("mango silver orchard flame", "timber quartz", "orchard flame"),
    ]
    for left_text, right_text, removed in cases:
        left = _grounded_trace(left_text, weak_words={2, 3}, weak_center=0.72)
        right = _grounded_trace(right_text)
        decision = _cross_segment_suffix_decision(
            left,
            right,
            _segment_for_cross(0, 0, 0, 500),
            _segment_for_cross(0, 1, 500, 1000),
            1000,
            500,
            500,
            Settings(),
        )
        assert decision is not None
        assert decision.removed_text == removed


def test_cross_segment_guard_keeps_visually_supported_tail():
    from government_ocr_text_api.vietocr_recognizer import _cross_segment_suffix_decision

    left = _grounded_trace("alpha bravo lumen sable", weak_words=set())
    right = _grounded_trace("charlie delta", weak_words=set())
    decision = _cross_segment_suffix_decision(
        left,
        right,
        _segment_for_cross(0, 0, 0, 500),
        _segment_for_cross(0, 1, 500, 1000),
        1000,
        500,
        500,
        Settings(),
    )
    assert decision is None


def test_cross_segment_numeric_tail_is_warning_only():
    from government_ocr_text_api.vietocr_recognizer import _cross_segment_suffix_decision

    left = _grounded_trace("alpha bravo 123 456", weak_words={2, 3}, weak_center=0.72)
    right = _grounded_trace("charlie delta", weak_words=set())
    decision = _cross_segment_suffix_decision(
        left,
        right,
        _segment_for_cross(0, 0, 0, 500),
        _segment_for_cross(0, 1, 500, 1000),
        1000,
        500,
        500,
        Settings(decoder_evidence_numeric_apply_changes=True),
    )
    assert decision is not None
    assert decision.numeric_only_warning is True


def test_1359_defaults_enable_selective_semantic_safe_verification(monkeypatch):
    monkeypatch.delenv("GOVERNMENT_OCR_SECONDARY_RECOGNIZER_ENABLED", raising=False)
    settings = Settings()
    assert settings.app_version == "1.3.5.9"
    assert settings.decoder_evidence_visual_grounding_enabled is True
    assert settings.decoder_evidence_cross_segment_enabled is True
    assert settings.performance_diagnostics_enabled is True
    assert settings.cpu_runtime_tuning_enabled is True
    assert settings.torch_cpu_threads == 4
    assert settings.paddle_cpu_threads == 4
    assert settings.paddle_enable_mkldnn is False
    assert settings.tesseract_verifier_enabled is True
    assert settings.tesseract_fail_closed is True
    assert settings.pad_batches_to_common_width is False
    assert settings.secondary_recognizer_enabled is True
    assert settings.secondary_recognizer_apply_changes is False
    assert settings.semantic_verification_enabled is True
    assert settings.semantic_selective_verification_enabled is True
    assert settings.semantic_auto_trim_enabled is True


def test_secondary_verifier_only_confirms_delete_only_proposal():
    from government_ocr_text_api.vietocr_recognizer import _secondary_prefers_deletion

    relation, proposed_ratio, raw_ratio = _secondary_prefers_deletion(
        "alpha bravo extra charlie delta",
        "alpha bravo charlie delta",
        "alpha bravo charlie delta",
        0.08,
    )
    assert relation == "primary_extra"
    assert proposed_ratio > raw_ratio


def test_secondary_verifier_conflict_never_rewrites_primary():
    from government_ocr_text_api.vietocr_recognizer import _secondary_prefers_deletion

    relation, proposed_ratio, raw_ratio = _secondary_prefers_deletion(
        "alpha bravo charlie delta",
        "alpha bravo delta",
        "alpha bravo charlie delta",
        0.08,
    )
    assert relation == "conflict"
    assert raw_ratio > proposed_ratio


def test_recognizer_applies_cross_segment_visual_gap_and_reports_metrics(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    left_text = "alpha bravo lumen sable"
    right_text = "charlie delta"
    left_trace = _grounded_trace(left_text, weak_words={2, 3}, weak_center=0.72)
    right_trace = _grounded_trace(right_text)
    left_segment = _segment_for_cross(0, 0, 0, 500)
    right_segment = _segment_for_cross(0, 1, 500, 1000)

    class TwoSegmentPredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            return [left_text, right_text], [0.55, 0.95]

    def fake_prepare(self, crops, predictor):
        return (
            [left_segment, right_segment],
            [[0, 1]],
            {
                "wide_crop_count": 1,
                "segment_count": 2,
                "split_segment_count": 1,
                "split_overlap_source_pixels": 0,
                "max_resized_width_before_split": 1000,
                "max_resized_width_after_split": 500,
                "width_cap_detected_count": 1,
                "width_cap_resolved_count": 1,
                "width_cap_unresolved_count": 0,
                "width_capped_after_split_count": 0,
            },
            set(),
        )

    monkeypatch.setattr(module.VietOcrRecognizer, "_prepare_segments", fake_prepare)
    monkeypatch.setattr(
        module,
        "_trace_seq2seq_attention_batch_detailed",
        lambda predictor, images, settings: [
            module._DecoderEvidenceTraceOutcome(trace=left_trace),
            module._DecoderEvidenceTraceOutcome(trace=right_trace),
        ],
    )
    polygon = LinePolygon([(0, 0), (1000, 0), (1000, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", Image.new("RGB", (1000, 32), "white"), polygon)
    recognizer = VietOcrRecognizer(
        Settings(
            decoder_evidence_full_coverage=True,
            hallucination_guard_enabled=False,
            secondary_recognizer_enabled=False,
        ),
        predictor=TwoSegmentPredictor(),
    )

    result = recognizer.recognize([crop])

    assert result[0].text == "alpha bravo charlie delta"
    metrics = recognizer.last_batch_metrics
    assert metrics["decoder_evidence_line_evidence_count"] == 1
    assert metrics["decoder_evidence_cross_segment_candidate_count"] == 1
    assert metrics["decoder_evidence_cross_segment_trimmed_count"] == 1
    assert any(
        event.get("reason") == "cross_segment_visual_gap"
        for event in recognizer.last_page_metrics[0]["decoder_evidence_events"]
    )


def test_scale_invariant_visual_grounding_detects_reuse_for_diffuse_attention():
    import numpy as np
    from government_ocr_text_api.vietocr_recognizer import _visual_grounding_step

    positions = np.arange(128, dtype=np.float32)
    first = np.exp(-0.5 * ((positions - 70.0) / 10.0) ** 2).astype(np.float32)
    first /= first.sum()
    repeated = np.exp(-0.5 * ((positions - 71.0) / 10.0) ** 2).astype(np.float32)
    repeated /= repeated.sum()
    assert float(first.max()) < 0.08  # reproduces the real-scale failure in 1.3.5.4

    ink = np.ones(128, dtype=np.float32)
    previous = np.zeros(128, dtype=np.float32)
    novel1, gain1, reuse1 = _visual_grounding_step(first, ink, previous)
    previous = np.maximum(previous, first)
    novel2, gain2, reuse2 = _visual_grounding_step(repeated, ink, previous)

    assert novel1 > 0.95
    assert reuse1 < 0.01
    assert reuse2 > 0.90
    assert novel2 < 0.10
    assert gain2 < gain1 * 0.10


def test_scale_invariant_visual_grounding_is_stable_across_encoder_lengths():
    import numpy as np
    from government_ocr_text_api.vietocr_recognizer import _visual_grounding_step

    reuse_values = []
    for length in (64, 128, 256):
        positions = np.arange(length, dtype=np.float32)
        sigma = length * 0.08
        first = np.exp(-0.5 * ((positions - length * 0.55) / sigma) ** 2).astype(np.float32)
        first /= first.sum()
        second = np.exp(-0.5 * ((positions - length * 0.56) / sigma) ** 2).astype(np.float32)
        second /= second.sum()
        previous = np.maximum(np.zeros(length, dtype=np.float32), first)
        _, _, reuse = _visual_grounding_step(second, np.ones(length, dtype=np.float32), previous)
        reuse_values.append(reuse)
    assert min(reuse_values) > 0.90
    assert max(reuse_values) - min(reuse_values) < 0.03


def test_decoder_trace_profiling_separates_model_and_visual_grounding(monkeypatch):
    from government_ocr_text_api.vietocr_recognizer import (
        _trace_seq2seq_attention_batch_detailed,
    )

    predictor, _ = _contract_faithful_trace_predictor(
        monkeypatch,
        token_ids=[4, 5],
        token_text="ab",
    )
    profile = {}
    outcomes = _trace_seq2seq_attention_batch_detailed(
        predictor,
        [_ink_image()],
        Settings(),
        profile=profile,
    )

    assert outcomes[0].trace is not None
    assert profile["decoder_forward_call_count"] >= 3  # a, b, EOS
    assert profile["decoder_sample_step_count"] >= 3
    assert profile["decoder_attention_element_count"] > 0
    assert profile["decoder_trace_input_pixel_count"] > 0
    assert profile["decoder_model_wall_ms"] >= 0.0
    assert profile["decoder_visual_grounding_wall_ms"] >= 0.0
    assert profile["decoder_trace_build_wall_ms"] >= 0.0


def test_performance_diagnostics_expose_stable_workload_fingerprint():
    polygon = LinePolygon([(0, 0), (400, 0), (400, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop("p0000-l0000", _synthetic_text_line(400), polygon),
        LineCrop("p0000-l0001", _synthetic_text_line(400), polygon),
    ]
    settings = Settings(
        decoder_evidence_enabled=False,
        hallucination_guard_enabled=False,
        performance_diagnostics_enabled=True,
    )
    recognizer = VietOcrRecognizer(settings, predictor=RecordingPredictor())

    recognizer.recognize(crops)
    first = dict(recognizer.last_batch_metrics["workload_fingerprint"])
    diagnostics = recognizer.last_batch_metrics["performance_diagnostics"]
    recognizer.recognize(crops)
    second = dict(recognizer.last_batch_metrics["workload_fingerprint"])

    assert first["fingerprint"] == second["fingerprint"]
    assert first["segment_count"] == second["segment_count"]
    assert diagnostics["recognizer_wall_ms"] >= 0.0
    assert diagnostics["recognizer_cpu_ms"] >= 0.0
    assert diagnostics["unaccounted_wall_ms"] >= 0.0
    assert "runtime_start" in diagnostics and "runtime_end" in diagnostics


def test_greedy_batch_fallback_is_visible_in_metrics():
    class FailingBatchPredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            raise RuntimeError("synthetic batch failure")

        def predict(self, image, return_prob=False):
            if return_prob:
                return "fallback text", 0.91
            return "fallback text"

    polygon = LinePolygon([(0, 0), (240, 0), (240, 32), (0, 32)], 1.0, "test", "test")
    crops = [
        LineCrop(
            f"p0000-l{i:04d}",
            _synthetic_text_line(240),
            polygon,
        )
        for i in range(3)
    ]
    recognizer = VietOcrRecognizer(
        Settings(
            decoder_evidence_enabled=False,
            split_wide_crops=False,
            recognition_batch_size=8,
        ),
        predictor=FailingBatchPredictor(),
    )

    result = recognizer.recognize(crops)

    assert [item.text for item in result] == ["fallback text"] * 3
    assert recognizer.last_batch_metrics["greedy_batch_fallback_count"] == 1
    assert recognizer.last_batch_metrics["greedy_batch_fallback_segment_count"] == 3
    assert recognizer.last_batch_metrics["greedy_batch_fallback_error_types"] == {"RuntimeError": 1}


def test_split_line_context_is_trace_only_not_local_mutation(monkeypatch):
    import government_ocr_text_api.vietocr_recognizer as module

    class LowConfidencePredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            self.batch_sizes.append(len(images))
            return ["alpha beta gamma"] * len(images), [0.20] * len(images)

    def trace_batch(predictor, images, settings, profile=None):
        text = "alpha beta gamma"
        trace = module._DecoderEvidenceTrace(
            raw_text=text,
            token_probabilities=tuple([0.90] * len(text)),
            attention_centers=tuple([0.5] * len(text)),
            attention_spreads=tuple([0.02] * len(text)),
            ink_support=tuple([0.4] * len(text)),
            last_ink_position=0.95,
        )
        return [module._DecoderEvidenceTraceOutcome(trace=trace) for _ in images]

    decision_calls = []

    def fake_decision(trace, settings):
        decision_calls.append(trace.raw_text)
        return None

    monkeypatch.setattr(module, "_trace_seq2seq_attention_batch_detailed", trace_batch)
    monkeypatch.setattr(module, "_decoder_evidence_decision", fake_decision)

    polygon = LinePolygon([(0, 0), (1200, 0), (1200, 32), (0, 32)], 1.0, "test", "test")
    crop = LineCrop("p0000-l0000", _synthetic_text_line(1200), polygon)
    recognizer = VietOcrRecognizer(
        Settings(
            decoder_evidence_enabled=True,
            decoder_evidence_full_coverage=False,
            decoder_evidence_max_checks_per_page=1,
            decoder_evidence_include_split_line_context=True,
            hallucination_guard_enabled=False,
        ),
        predictor=LowConfidencePredictor(),
    )

    recognizer.recognize([crop])

    assert recognizer.last_batch_metrics["decoder_evidence_seed_selected_count"] == 1
    assert recognizer.last_batch_metrics["decoder_evidence_context_forced_count"] >= 1
    assert recognizer.last_batch_metrics["decoder_evidence_trace_count"] > 1
    assert len(decision_calls) == 1


def test_common_width_padding_is_disabled_by_default():
    assert Settings().pad_batches_to_common_width is False


class FakeSecondaryResult:
    def __init__(self, text: str, confidence: float):
        self.json = {"res": {"rec_text": text, "rec_score": confidence}}


class RecordingSecondaryModel:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def predict(self, *, input, batch_size):
        self.calls.append((list(input), batch_size))
        return list(self.values)


def test_semantic_verification_applies_consensus_split_retry_and_keeps_raw_audit():
    from government_ocr_text_api.semantic_retry import RetryVariant

    original = "cho nguy nguy nguy nhi phương có tỷ lệ điều tiết"
    verifier = "c) 50% nhu cầu kinh phí tăng thêm cho địa phương có tỷ lệ điều tiết"

    seen_retry_crops = []

    class RetryStubRecognizer(VietOcrRecognizer):
        def _semantic_retry_variants(self, crop):
            seen_retry_crops.append(crop)
            return (
                RetryVariant(
                    "c) 50% nhu cầu kinh phi tăng thêm cho địa phương "
                    "có tỷ lệ điều tiết",
                    0.88,
                    320,
                ),
            )

    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(900),
        LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = RetryStubRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [
            FakeSecondaryResult(
                "c) 50% nhu cu kinh phí tăng thêm cho đa phương có t l điu tit",
                0.97,
            )
        ]
    )
    metrics = {0: {}}
    retry_crop = LineCrop(
        crop.crop_id,
        _synthetic_text_line(960),
        crop.polygon,
    )

    result = recognizer._apply_semantic_verification(
        [crop],
        [
            Recognition(
                original,
                0.34,
                "decoder_loop_trimmed",
                semantic_risk="high",
                semantic_reasons=("primary_recognition_risk", "tesseract_numeric_disagreement"),
                verifier_text=verifier,
                verifier_confidence=0.96,
            )
        ],
        metrics,
        retry_crops=[retry_crop],
    )

    assert result[0].text == verifier
    assert result[0].raw_text == original
    assert result[0].error_code is None
    assert result[0].semantic_risk == "medium"
    assert result[0].semantic_reasons == ("consensus_split_retry_applied",)
    assert result[0].secondary_confidence == 0.97
    assert metrics[0]["semantic_consensus_retry_count"] == 1
    assert metrics[0]["semantic_consensus_retry_eligible_count"] == 1
    retry_events = metrics[0]["semantic_consensus_retry_events"]
    assert len(retry_events) == 1
    assert retry_events[0] == {
        "crop_id": "p0000-l0000",
        "applied": True,
        "reason": "consensus_split_retry_applied",
        "selected_width": 320,
        "confidence": 0.88,
        "elapsed_ms": retry_events[0]["elapsed_ms"],
        "before_text": original,
        "after_text": verifier,
        "primary_confidence": 0.34,
        "verifier_text": verifier,
        "verifier_confidence": 0.96,
        "secondary_text": (
            "c) 50% nhu cu kinh phí tăng thêm cho đa phương có t l điu tit"
        ),
        "secondary_confidence": 0.97,
        "variants": [
            {
                "text": (
                    "c) 50% nhu cầu kinh phi tăng thêm cho địa phương "
                    "có tỷ lệ điều tiết"
                ),
                "confidence": 0.88,
                "resized_width": 320,
            }
        ],
    }
    assert retry_events[0]["elapsed_ms"] >= 0
    assert seen_retry_crops == [retry_crop]


def test_partial_remediation_can_repair_high_confidence_line_with_three_engine_consensus():
    from government_ocr_text_api.semantic_retry import RetryVariant

    original = "Điều 16 có hiệu lực từ ngày 01 tháng 01 năm 2026"
    expected = "Điều 15 có hiệu lực từ ngày 01 tháng 01 năm 2026"

    class RemediationStubRecognizer(VietOcrRecognizer):
        def _semantic_retry_variants(self, crop):
            return (RetryVariant(expected, 0.92, 320),)

    crop = LineCrop(
        "p0000-l0002",
        _synthetic_text_line(900),
        LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = RemediationStubRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Dieu 15 co hieu luc tu ngay 01 thang 01 nam 2026", 0.97)]
    )
    metrics = {0: {}}

    result = recognizer.remediate_high_risk_candidates(
        [crop],
        [
            Recognition(
                original,
                0.94,
                semantic_risk="high",
                semantic_reasons=("tesseract_numeric_disagreement",),
                verifier_text=expected,
                verifier_confidence=0.98,
            )
        ],
        metrics,
    )

    assert result[0].text == expected
    assert result[0].raw_text == original
    assert result[0].semantic_risk == "medium"
    assert metrics[0]["attempted_count"] == 1
    assert metrics[0]["applied_count"] == 1
    assert metrics[0]["events"][0]["before_text"] == original
    assert metrics[0]["events"][0]["after_text"] == expected


def test_partial_remediation_rejects_numeric_disagreement_and_keeps_high_risk():
    from government_ocr_text_api.semantic_retry import RetryVariant

    original = "Điều 16 có hiệu lực"
    verifier = "Điều 15 có hiệu lực"

    class RemediationStubRecognizer(VietOcrRecognizer):
        def _semantic_retry_variants(self, crop):
            return (RetryVariant(verifier, 0.93, 320),)

    crop = LineCrop(
        "p0000-l0003",
        _synthetic_text_line(600),
        LinePolygon([(0, 0), (600, 0), (600, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = RemediationStubRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Dieu 16 co hieu luc", 0.98)]
    )
    metrics = {0: {}}

    result = recognizer.remediate_high_risk_candidates(
        [crop],
        [
            Recognition(
                original,
                0.94,
                semantic_risk="high",
                semantic_reasons=("tesseract_numeric_disagreement",),
                verifier_text=verifier,
                verifier_confidence=0.98,
            )
        ],
        metrics,
    )

    assert result[0].text == original
    assert result[0].semantic_risk == "high"
    assert metrics[0]["applied_count"] == 0
    assert metrics[0]["events"][0]["reason"] == "consensus_retry_numeric_disagreement"


def test_semantic_retry_variants_preserve_sequential_output_and_reuse_request_cache():
    class SequentialRecordingPredictor(RecordingPredictor):
        def predict(self, image, return_prob=False):
            self.predict_calls += 1
            return "đoạn văn", 0.95

    polygon = LinePolygon(
        [(0, 0), (900, 0), (900, 32), (0, 32)],
        1.0,
        "test",
        "test",
    )
    crop = LineCrop("p0000-l0000", _synthetic_text_line(900), polygon)
    predictor = SequentialRecordingPredictor()
    recognizer = VietOcrRecognizer(Settings(), predictor=predictor)

    first = recognizer._semantic_retry_variants_many([crop])
    calls_after_first = predictor.predict_calls
    duplicate = LineCrop("p0000-l9999", crop.image.copy(), polygon)
    second = recognizer._semantic_retry_variants_many([duplicate])

    assert [[variant.resized_width for variant in values] for values in first] == [
        [288, 320, 384],
    ]
    assert [[variant.text for variant in values] for values in first] == [
        ["đoạn văn", "đoạn văn", "đoạn văn"],
    ]
    assert second == first
    assert calls_after_first > 0
    assert predictor.batch_sizes == []
    assert predictor.predict_calls == calls_after_first


def test_semantic_retry_budget_is_assigned_after_eligibility_and_secondary_omission():
    from government_ocr_text_api.semantic_retry import RetryVariant

    corrupt = "tỷ lệ phân bổ từ ngân sách trung ương, tỷ lệ"
    expected = "c) Xác định tỷ lệ phân bổ vốn hỗ trợ từ ngân sách trung ương, tỷ lệ"
    seen_retry_ids = []

    class RetryStubRecognizer(VietOcrRecognizer):
        def _semantic_retry_variants(self, crop):
            seen_retry_ids.append(crop.crop_id)
            return (RetryVariant(expected, 0.91, 320),)

    crops = [
        LineCrop(
            f"p0000-l{index:04d}",
            _synthetic_text_line(900),
            LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
        )
        for index in range(3)
    ]
    recognizer = RetryStubRecognizer(
        Settings(
            secondary_recognizer_enabled=True,
            semantic_retry_max_lines_per_page=1,
        ),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [
            FakeSecondaryResult("Điều 13", 0.97),
            FakeSecondaryResult("không dùng", 0.97),
            FakeSecondaryResult(
                "c) Xac dinh ty le phan bo von ho tro tu ngan sach trung uong, ty le",
                0.97,
            ),
        ]
    )
    recognitions = [
        Recognition(
            "Điều 12",
            0.40,
            semantic_risk="high",
            semantic_reasons=("tesseract_numeric_disagreement",),
            verifier_text="Điều 13",
            verifier_confidence=0.96,
        ),
        Recognition(
            "Dòng không đủ điều kiện",
            0.40,
            semantic_risk="high",
            semantic_reasons=("tesseract_numeric_disagreement",),
            verifier_text="Dòng không đủ điều kiện",
            verifier_confidence=0.20,
        ),
        Recognition(
            corrupt,
            0.34,
            None,
            semantic_risk="high",
            semantic_reasons=("primary_recognition_risk", "tesseract_material_disagreement"),
            verifier_text=expected,
            verifier_confidence=0.96,
        ),
    ]
    metrics = {0: {}}

    result = recognizer._apply_semantic_verification(crops, recognitions, metrics)

    # The strong verifier/secondary wording now repairs line 2 directly, so
    # the remaining retry budget stays available for the numeric disagreement.
    assert seen_retry_ids == ["p0000-l0000"]
    assert result[2].text == expected
    assert metrics[0]["semantic_consensus_retry_attempted_count"] == 1
    assert metrics[0]["semantic_consensus_retry_count"] == 0
    assert metrics[0]["semantic_consensus_retry_eligible_count"] == 1
    assert metrics[0]["semantic_verifier_consensus_count"] == 1


def test_semantic_retry_budget_prioritizes_numeric_and_material_corruption():
    crops = [
        LineCrop(
            f"p0000-l{index:04d}",
            _synthetic_text_line(700),
            LinePolygon([(0, 0), (700, 0), (700, 32), (0, 32)], 1.0, "test", "test"),
        )
        for index in range(4)
    ]
    recognitions = [
        Recognition(
            "Dòng khác dấu thứ nhất",
            0.40,
            semantic_risk="high",
            semantic_reasons=("tesseract_diacritic_disagreement",),
        ),
        Recognition(
            "Dòng khác dấu thứ hai",
            0.39,
            semantic_risk="high",
            semantic_reasons=("tesseract_diacritic_disagreement",),
        ),
        Recognition(
            "Nội dung mất 50 phần trăm",
            0.45,
            semantic_risk="high",
            semantic_reasons=("tesseract_numeric_disagreement",),
        ),
        Recognition(
            "Nội dung bị mất một cụm từ",
            0.44,
            semantic_risk="high",
            semantic_reasons=("secondary_indicates_primary_omission",),
        ),
    ]

    selected = _prioritize_semantic_retry_indices(
        crops,
        recognitions,
        candidate_indices=range(4),
        max_per_page=2,
    )

    assert selected == {2, 3}


def test_semantic_verification_normalizes_legal_collocation_outside_candidates():
    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(700),
        LinePolygon([(0, 0), (700, 0), (700, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(Settings(), predictor=RecordingPredictor())
    metrics = {0: {}}

    result = recognizer._apply_semantic_verification(
        [crop],
        [Recognition("6.Bồ sung Điều 29a sau Điều 29 như sau:", 0.91)],
        metrics,
        candidate_indices=[],
    )

    assert result[0].text == "6.Bổ sung Điều 29a sau Điều 29 như sau:"
    assert result[0].raw_text == "6.Bồ sung Điều 29a sau Điều 29 như sau:"
    assert result[0].semantic_risk == "medium"
    assert result[0].semantic_reasons == ("legal_collocation_normalized",)


def test_semantic_verification_applies_three_engine_separator_consensus_without_retry():
    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(900),
        LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Nam; s, ngày cp Giy chứng nhận đăng ký doanh nghiệp", 0.97)]
    )
    metrics = {0: {}}

    result = recognizer._apply_semantic_verification(
        [crop],
        [
            Recognition(
                "Nam, số, ngày cấp Giấy chứng nhận đăng ký doanh nghiệp",
                0.92,
                verifier_text=(
                    "Nam; sô, ngày cap Giây chứng nhận đăng ký doanh nghiệp"
                ),
                verifier_confidence=0.96,
            )
        ],
        metrics,
    )

    assert result[0].text == "Nam; số, ngày cấp Giấy chứng nhận đăng ký doanh nghiệp"
    assert result[0].raw_text == "Nam, số, ngày cấp Giấy chứng nhận đăng ký doanh nghiệp"
    assert result[0].semantic_reasons == (
        "three_engine_separator_consensus_applied",
    )
    assert metrics[0]["semantic_surface_consensus_count"] == 1


def test_separator_consensus_revalidates_and_clears_stale_tesseract_numeric_risk():
    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(700),
        LinePolygon([(0, 0), (700, 0), (700, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Dieu 5. Nguon von thuc hien", 0.97)]
    )

    result = recognizer._apply_semantic_verification(
        [crop],
        [
            Recognition(
                "Điều 5 Nguồn vốn thực hiện",
                0.92,
                semantic_risk="high",
                semantic_reasons=("tesseract_numeric_disagreement",),
                verifier_text="Điều 5. Nguồn vốn thực hiện",
                verifier_confidence=0.96,
            )
        ],
        {0: {}},
    )

    assert result[0].text == "Điều 5. Nguồn vốn thực hiện"
    assert result[0].semantic_risk == "medium"
    assert result[0].semantic_reasons == (
        "three_engine_separator_consensus_applied",
    )


def test_separator_consensus_revalidation_keeps_real_tesseract_diacritic_risk():
    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(700),
        LinePolygon([(0, 0), (700, 0), (700, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Dieu 5. Nguon von thuc hien", 0.97)]
    )

    result = recognizer._apply_semantic_verification(
        [crop],
        [
            Recognition(
                "Điều 5 Nguồn vốn thực hiện",
                0.92,
                semantic_risk="high",
                semantic_reasons=("tesseract_numeric_disagreement",),
                verifier_text="Điều 5. Nguôn vốn thực hiện",
                verifier_confidence=0.96,
            )
        ],
        {0: {}},
    )

    assert result[0].text == "Điều 5. Nguồn vốn thực hiện"
    assert result[0].semantic_risk == "high"
    assert result[0].semantic_reasons == (
        "three_engine_separator_consensus_applied",
        "tesseract_diacritic_disagreement",
    )


def test_semantic_verification_adopts_two_engine_word_repair_without_vietocr_retry():
    primary = "?Điều 5. Nguyên tắc thông ký website thương mại điện tử"
    verifier = "“Điều 5. Nguyên tắc thông báo, đăng ký website thương mại điện tử"
    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(900),
        LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [
            FakeSecondaryResult(
                "Điu 5. Nguyên tc thông báo, đăng ký website thương mi đin t",
                0.97,
            )
        ]
    )
    metrics = {0: {}}

    result = recognizer._apply_semantic_verification(
        [crop],
        [
            Recognition(
                primary,
                0.86,
                semantic_risk="high",
                semantic_reasons=("tesseract_material_disagreement",),
                verifier_text=verifier,
                verifier_confidence=0.94,
            )
        ],
        metrics,
    )

    assert result[0].text == verifier
    assert result[0].raw_text == primary
    assert result[0].semantic_risk == "medium"
    assert result[0].semantic_reasons == ("verifier_secondary_consensus_applied",)
    assert metrics[0]["semantic_verifier_consensus_count"] == 1


def test_low_confidence_primary_can_use_strong_two_engine_word_consensus():
    primary = "thống nhất từ trung đơn địa phương"
    verifier = "thống nhất từ trung ương đến địa phương"
    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(900),
        LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("thong nhat tu trung uong den dia phuong", 0.98)]
    )
    metrics = {0: {}}

    result = recognizer._apply_semantic_verification(
        [crop],
        [
            Recognition(
                primary,
                0.70,
                semantic_risk="high",
                semantic_reasons=("tesseract_material_disagreement",),
                verifier_text=verifier,
                verifier_confidence=0.97,
            )
        ],
        metrics,
    )

    assert result[0].text == verifier
    assert result[0].raw_text == primary
    assert result[0].semantic_reasons == ("verifier_secondary_consensus_applied",)
    assert metrics[0]["semantic_verifier_consensus_count"] == 1


def test_semantic_secondary_batch_trims_suffix_and_keeps_raw_text():
    original = (
        "59/2015/TT-BCT Ngày 31 tháng 12 năm 2015 của Bộ Công "
        "Thương quy định về người thuy nhi viện thuy nghiệp thuyên thiến"
    )
    crops = [
        LineCrop(
            "p0000-l0000",
            _synthetic_text_line(900),
            LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
        ),
        LineCrop(
            "p0000-l0001",
            _synthetic_text_line(500),
            LinePolygon([(0, 40), (500, 40), (500, 72), (0, 72)], 1.0, "test", "test"),
        ),
    ]
    secondary = RecordingSecondaryModel(
        [
            FakeSecondaryResult(
                "59/2015/TT-BCT ngày 31 tháng 12 năm 2015 ca B Công Thưng quy đnh v",
                0.97,
            ),
            FakeSecondaryResult("xâm phm an ninh quc gia, khng b", 0.98),
        ]
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = secondary
    metrics = {0: {}}

    results = recognizer._apply_semantic_verification(
        crops,
        [
            Recognition(original, 0.47, "decoder_loop_trimmed"),
            Recognition("xâm phạm ninh quốc gia, khủng bố", 0.85),
        ],
        metrics,
    )

    assert len(secondary.calls) == 1
    assert len(secondary.calls[0][0]) == 2
    assert secondary.calls[0][1] == 32
    assert results[0].text.endswith("quy định về")
    assert results[0].raw_text == original
    assert results[0].semantic_risk == "medium"
    assert results[0].semantic_reasons == ("unsupported_suffix_removed",)
    assert results[1].text == "xâm phạm ninh quốc gia, khủng bố"
    assert results[1].semantic_risk == "high"
    assert results[1].semantic_reasons == ("secondary_indicates_primary_omission",)
    assert metrics[0]["semantic_verified_count"] == 2
    assert metrics[0]["semantic_auto_trimmed_count"] == 1
    assert metrics[0]["semantic_high_risk_count"] == 1


def test_semantic_secondary_failure_preserves_primary_and_marks_risk():
    class FailingSecondaryModel:
        def predict(self, *, input, batch_size):
            raise RuntimeError("secondary failed")

    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(500),
        LinePolygon([(0, 0), (500, 0), (500, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = FailingSecondaryModel()
    metrics = {0: {}}

    results = recognizer._apply_semantic_verification(
        [crop],
        [Recognition("Nội dung chưa chắc chắn", 0.50, "tail_segment_uncertain")],
        metrics,
    )

    assert results[0].text == "Nội dung chưa chắc chắn"
    assert results[0].semantic_risk == "high"
    assert results[0].semantic_reasons == ("secondary_unavailable",)
    assert metrics[0]["semantic_secondary_error_count"] == 1


def test_selective_semantic_verification_batches_only_candidates_and_preserves_prior_risk():
    crops = [
        LineCrop(
            f"p0000-l{index:04d}",
            _synthetic_text_line(500),
            LinePolygon(
                [(0, index * 40), (500, index * 40), (500, index * 40 + 32), (0, index * 40 + 32)],
                1.0,
                "test",
                "test",
            ),
        )
        for index in range(3)
    ]
    secondary = RecordingSecondaryModel(
        [FakeSecondaryResult("Điều 44", 0.98)]
    )
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=True,
            semantic_selective_verification_enabled=True,
        ),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = secondary
    metrics = {0: {}}
    recognitions = [
        Recognition("Dòng an toàn thứ nhất", 0.97),
        Recognition(
            "Điều 14",
            0.96,
            semantic_risk="high",
            semantic_reasons=("tesseract_numeric_disagreement",),
            verifier_text="Điều 44",
            verifier_confidence=0.97,
        ),
        Recognition("Dòng an toàn thứ ba", 0.96),
    ]

    result = recognizer.verify_semantic_candidates(
        crops,
        recognitions,
        [1],
        metrics,
    )

    assert len(secondary.calls) == 1
    assert len(secondary.calls[0][0]) == 1
    assert result[0] is recognitions[0]
    assert result[2] is recognitions[2]
    assert result[1].text == "Điều 14"
    assert result[1].semantic_risk == "high"
    assert "tesseract_numeric_disagreement" in result[1].semantic_reasons
    assert "secondary_numeric_disagreement" in result[1].semantic_reasons
    assert result[1].verifier_text == "Điều 44"
    assert metrics[0]["semantic_candidate_count"] == 1
    assert metrics[0]["semantic_skipped_count"] == 2
    assert metrics[0]["semantic_verified_count"] == 1


def test_secondary_recognizer_retries_once_without_mkldnn_on_pir_error(monkeypatch):
    class PirFailingSecondaryModel:
        def predict(self, *, input, batch_size):
            raise NotImplementedError(
                "ConvertPirAttribute2RuntimeAttribute not support "
                "[pir::ArrayAttribute<pir::DoubleAttribute>] at onednn_instruction.cc"
            )

    safe_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Ná»™i dung an toÃ n", 0.98)]
    )
    constructor_calls = []

    def fake_text_recognition(**kwargs):
        constructor_calls.append(kwargs)
        return safe_model

    import paddleocr

    monkeypatch.setattr(paddleocr, "TextRecognition", fake_text_recognition)
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=True,
            paddle_enable_mkldnn=True,
            paddle_mkldnn_fallback=True,
        ),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = PirFailingSecondaryModel()
    recognizer._secondary_mkldnn_effective = True

    result = recognizer._secondary_recognize_lines([_synthetic_text_line(500)])

    assert result[0] is not None
    assert result[0].confidence == 0.98
    assert constructor_calls[0]["enable_mkldnn"] is False
    assert recognizer._secondary_fallback_used is True
    assert len(safe_model.calls) == 1


def test_secondary_malformed_score_is_isolated_as_unavailable():
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=RecordingPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [FakeSecondaryResult("Nội dung", "bad-score")]
    )

    result = recognizer._secondary_recognize_lines([_synthetic_text_line(500)])

    assert result == [None]
    assert "secondary_parse_error" in recognizer._secondary_model_error


def test_targeted_recognition_is_semantically_verified_before_return():
    original = (
        "59/2015/TT-BCT Ngày 31 tháng 12 năm 2015 của Bộ Công "
        "Thương quy định về người thuy nhi viện thuy nghiệp thuyên thiến"
    )

    class TargetedPredictor(RecordingPredictor):
        def predict(self, image, return_prob=False):
            return original, 0.47

    crop = LineCrop(
        "p0000-l0000-chapter-retry",
        _synthetic_text_line(900),
        LinePolygon([(0, 0), (900, 0), (900, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(secondary_recognizer_enabled=True),
        predictor=TargetedPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [
            FakeSecondaryResult(
                "59/2015/TT-BCT ngày 31 tháng 12 năm 2015 ca B Công Thưng quy đnh v",
                0.97,
            )
        ]
    )

    result = recognizer.recognize_targeted([crop])

    assert result[0].text.endswith("quy định về")
    assert result[0].raw_text == original
    assert result[0].semantic_risk == "medium"
    assert result[0].semantic_reasons == ("unsupported_suffix_removed",)


def test_recognize_runs_semantic_verification_on_final_primary_result():
    original = (
        "59/2015/TT-BCT Ngày 31 tháng 12 năm 2015 của Bộ Công "
        "Thương quy định về người thuy nhi viện thuy nghiệp thuyên thiến"
    )

    class PrimaryPredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            return [original] * len(images), [0.47] * len(images)

    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(400),
        LinePolygon([(0, 0), (400, 0), (400, 32), (0, 32)], 1.0, "test", "test"),
    )
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=True,
            semantic_selective_verification_enabled=False,
            decoder_evidence_enabled=False,
            hallucination_guard_enabled=False,
            split_wide_crops=False,
            beam_retry_enabled=False,
        ),
        predictor=PrimaryPredictor(),
    )
    recognizer._secondary_model = RecordingSecondaryModel(
        [
            FakeSecondaryResult(
                "59/2015/TT-BCT ngày 31 tháng 12 năm 2015 ca B Công Thưng quy đnh v",
                0.97,
            )
        ]
    )

    result = recognizer.recognize([crop])

    assert result[0].text.endswith("quy định về")
    assert result[0].raw_text == original
    assert recognizer.last_page_metrics[0]["semantic_auto_trimmed_count"] == 1


def test_recognize_defers_secondary_batch_when_selective_verification_is_enabled():
    class PrimaryPredictor(RecordingPredictor):
        def predict_batch(self, images, return_prob=True):
            return ["Nội dung rõ ràng"] * len(images), [0.97] * len(images)

    crop = LineCrop(
        "p0000-l0000",
        _synthetic_text_line(400),
        LinePolygon([(0, 0), (400, 0), (400, 32), (0, 32)], 1.0, "test", "test"),
    )
    secondary = RecordingSecondaryModel(
        [FakeSecondaryResult("Nội dung rõ ràng", 0.98)]
    )
    recognizer = VietOcrRecognizer(
        Settings(
            secondary_recognizer_enabled=True,
            semantic_selective_verification_enabled=True,
            decoder_evidence_enabled=False,
            hallucination_guard_enabled=False,
            split_wide_crops=False,
            beam_retry_enabled=False,
        ),
        predictor=PrimaryPredictor(),
    )
    recognizer._secondary_model = secondary

    result = recognizer.recognize([crop])

    assert result[0].text == "Nội dung rõ ràng"
    assert secondary.calls == []
    assert recognizer.last_page_metrics[0]["semantic_deferred_count"] == 1
