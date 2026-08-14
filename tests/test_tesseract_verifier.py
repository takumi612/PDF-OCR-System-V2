import threading

from PIL import Image

from government_ocr_text_api.config import Settings
from government_ocr_text_api.models import (
    BBox,
    LineCrop,
    LinePolygon,
    PageImage,
    Recognition,
)
from government_ocr_text_api.tesseract_verifier import (
    TesseractLine,
    TesseractPageVerifier,
    apply_tesseract_verification,
    match_tesseract_lines,
    parse_tesseract_tsv,
)


def test_parser_keeps_rows_when_recognized_text_starts_with_quote():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t40\t12\t96.0\t\"quả\n"
        "5\t1\t1\t1\t1\t2\t55\t20\t30\t12\t94.0\tđịnh\n"
        "5\t1\t1\t1\t2\t1\t10\t40\t50\t12\t92.0\tĐiều\n"
    )

    lines = parse_tesseract_tsv(tsv)

    assert [line.text for line in lines] == ['"quả định', "Điều"]
    assert lines[0].bbox == BBox(10.0, 20.0, 85.0, 32.0)
    assert round(lines[0].confidence, 4) == 0.95


def test_matcher_uses_page_geometry_not_result_order():
    crop_boxes = [
        BBox(10, 10, 500, 35),
        BBox(10, 50, 500, 75),
    ]
    tesseract_lines = [
        TesseractLine("Dòng thứ hai", 0.96, BBox(12, 51, 490, 74)),
        TesseractLine("Dòng thứ nhất", 0.97, BBox(12, 11, 490, 34)),
    ]

    matches = match_tesseract_lines(crop_boxes, tesseract_lines, min_score=0.55)

    assert [match.text if match else None for match in matches] == [
        "Dòng thứ nhất",
        "Dòng thứ hai",
    ]


def test_high_confidence_diacritic_disagreement_is_masked_without_rewrite():
    primary = Recognition("Cơ quan có thẩm quyền", 0.96)
    verifier = TesseractLine(
        "Cơ quan có thẩm quyên",
        0.95,
        BBox(0, 0, 100, 20),
    )

    result = apply_tesseract_verification(primary, verifier, Settings())

    assert result.text == primary.text
    assert result.semantic_risk == "high"
    assert "tesseract_diacritic_disagreement" in result.semantic_reasons
    assert result.verifier_text == verifier.text
    assert result.verifier_confidence == verifier.confidence


def test_numeric_disagreement_is_masked_without_rewrite():
    primary = Recognition("Điều 15 có hiệu lực từ năm 2026", 0.97)
    verifier = TesseractLine(
        "Điều 16 có hiệu lực từ năm 2026",
        0.96,
        BBox(0, 0, 100, 20),
    )

    result = apply_tesseract_verification(primary, verifier, Settings())

    assert result.text == primary.text
    assert result.semantic_risk == "high"
    assert "tesseract_numeric_disagreement" in result.semantic_reasons


def test_low_confidence_or_missing_verifier_never_downgrades_primary_risk():
    primary = Recognition(
        "Điều khoản chưa chắc chắn",
        0.55,
        semantic_risk="high",
        semantic_reasons=("primary_recognition_risk",),
    )
    low_confidence = TesseractLine(
        "Điều khoản chưa chắc chắn",
        0.20,
        BBox(0, 0, 100, 20),
    )

    with_low_confidence = apply_tesseract_verification(
        primary,
        low_confidence,
        Settings(),
    )
    without_verifier = apply_tesseract_verification(primary, None, Settings())

    assert with_low_confidence.semantic_risk == "high"
    assert with_low_confidence.semantic_reasons == ("primary_recognition_risk",)
    assert without_verifier.semantic_risk == "high"


def test_unavailable_required_verifier_fails_closed_without_losing_primary(tmp_path):
    settings = Settings(
        tesseract_executable_path=tmp_path / "missing-tesseract.exe",
        tesseract_data_path=tmp_path / "missing-tessdata",
    )
    verifier = TesseractPageVerifier(settings)
    page = PageImage(0, Image.new("RGB", (200, 100), "white"))
    crop = LineCrop(
        "p0000-l0000",
        Image.new("RGB", (180, 20), "white"),
        LinePolygon(
            [(10, 10), (190, 10), (190, 30), (10, 30)],
            0.99,
            "test",
            "test",
        ),
    )
    primary = Recognition("Điều 1. Phạm vi điều chỉnh", 0.97)

    results, metrics = verifier.verify(page, [crop], [primary])

    assert results[0].text == primary.text
    assert results[0].semantic_risk == "high"
    assert results[0].semantic_reasons == ("tesseract_verifier_unavailable",)
    assert metrics["status"] == "unavailable"


def test_unexpected_runtime_error_fails_closed_instead_of_crashing_request(
    tmp_path,
    monkeypatch,
):
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"placeholder")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "vie.traineddata").write_bytes(b"placeholder")
    (tessdata / "eng.traineddata").write_bytes(b"placeholder")
    verifier = TesseractPageVerifier(
        Settings(
            tesseract_executable_path=executable,
            tesseract_data_path=tessdata,
        )
    )
    monkeypatch.setattr(
        verifier,
        "_run_tsv",
        lambda page: (_ for _ in ()).throw(ValueError("malformed runtime output")),
    )
    page = PageImage(0, Image.new("RGB", (200, 100), "white"))
    primary = Recognition("Điều 1. Phạm vi điều chỉnh", 0.97)

    results, metrics = verifier.verify(page, [], [primary])

    assert results[0].text == primary.text
    assert results[0].semantic_risk == "high"
    assert results[0].semantic_reasons == ("tesseract_verifier_failed",)
    assert metrics["status"] == "failed"
    assert metrics["error_type"] == "ValueError"


def test_targeted_verifier_runs_single_line_psm_and_keeps_per_line_audit(
    tmp_path,
    monkeypatch,
):
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"placeholder")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "vie.traineddata").write_bytes(b"placeholder")
    (tessdata / "eng.traineddata").write_bytes(b"placeholder")
    verifier = TesseractPageVerifier(
        Settings(
            tesseract_executable_path=executable,
            tesseract_data_path=tessdata,
            partial_remediation_tesseract_psm=7,
        )
    )
    seen_psm = []
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t0\t0\t45\t18\t97.0\tĐiều\n"
        "5\t1\t1\t1\t1\t2\t50\t0\t20\t18\t96.0\t15\n"
    )

    def run_image_tsv(image, *, psm):
        seen_psm.append(psm)
        return tsv

    monkeypatch.setattr(verifier, "_run_image_tsv", run_image_tsv)
    crop = LineCrop(
        "p0000-l0004",
        Image.new("RGB", (180, 24), "white"),
        LinePolygon(
            [(10, 40), (190, 40), (190, 64), (10, 64)],
            0.99,
            "test",
            "test",
        ),
    )

    evidence, metrics = verifier.recognize_targeted([crop])

    assert seen_psm == [7]
    assert evidence[0] is not None
    assert evidence[0].text == "Điều 15"
    assert round(evidence[0].confidence, 3) == 0.967
    assert evidence[0].bbox == crop.polygon.bbox
    assert metrics["status"] == "complete"
    assert metrics["attempted_count"] == 1
    assert metrics["matched_count"] == 1
    assert metrics["events"][0]["crop_id"] == "p0000-l0004"
    assert metrics["events"][0]["psm"] == 7


def test_targeted_verifier_runs_independent_crops_concurrently(tmp_path, monkeypatch):
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"placeholder")
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "vie.traineddata").write_bytes(b"placeholder")
    (tessdata / "eng.traineddata").write_bytes(b"placeholder")
    verifier = TesseractPageVerifier(
        Settings(
            tesseract_executable_path=executable,
            tesseract_data_path=tessdata,
        )
    )
    barrier = threading.Barrier(2)
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t0\t0\t45\t18\t97.0\tĐiều\n"
    )

    def run_image_tsv(image, *, psm):
        barrier.wait(timeout=0.5)
        return tsv

    monkeypatch.setattr(verifier, "_run_image_tsv", run_image_tsv)
    polygon = LinePolygon(
        [(10, 40), (190, 40), (190, 64), (10, 64)],
        0.99,
        "test",
        "test",
    )
    crops = [
        LineCrop(f"p0000-l{index:04d}", Image.new("RGB", (180, 24), "white"), polygon)
        for index in range(2)
    ]

    evidence, metrics = verifier.recognize_targeted(crops)

    assert [item.text if item else None for item in evidence] == ["Điều", "Điều"]
    assert metrics["matched_count"] == 2
    assert [event["crop_id"] for event in metrics["events"]] == [
        "p0000-l0000",
        "p0000-l0001",
    ]
