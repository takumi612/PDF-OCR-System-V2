import warnings

from government_ocr_text_api.logging_utils import configure_logging, format_event


def test_compact_page_log_keeps_actionable_metrics_and_omits_nested_json():
    rendered = format_event(
        "page_complete",
        "Đã xử lý xong trang",
        log_format="compact",
        page_index=0,
        page_count=9,
        source="ocr",
        line_count=39,
        needs_review=True,
        metrics={
            "page_total_ms": 40590.674,
            "tesseract_verifier": {
                "input_line_count": 39,
                "matched_line_count": 35,
                "disagreement_line_count": 20,
            },
            "recognition": {
                "semantic_high_risk_count": 20,
                "semantic_consensus_retry_count": 2,
                "semantic_surface_consensus_count": 7,
                "semantic_verifier_consensus_count": 6,
                "decoder_evidence_events": [{"large": "diagnostic payload"}],
            },
        },
    )

    assert rendered == (
        "[OCR] Trang 1/9 | 39 dòng | 40.6s | cần duyệt: CÓ | "
        "rủi ro: 20 | Tesseract: 35/39, lệch 20 | sửa: retry=2, dấu=7, từ=6"
    )
    assert "diagnostic payload" not in rendered


def test_json_log_format_preserves_full_structured_payload():
    rendered = format_event(
        "document_complete",
        "Đã hoàn tất bóc tách PDF",
        log_format="json",
        filename="30-ttg.signed.pdf",
        status="partial",
        processing_time_ms=318994.228,
        semantic_risk_count=136,
        ai_ready=False,
    )

    assert '"event": "document_complete"' in rendered
    assert '"semantic_risk_count": 136' in rendered
    assert '"ai_ready": false' in rendered


def test_compact_page_log_explains_stage_timings_and_each_retry_decision():
    rendered = format_event(
        "page_complete",
        "Đã xử lý xong trang",
        log_format="compact",
        page_index=0,
        page_count=4,
        source="ocr",
        line_count=41,
        needs_review=True,
        metrics={
            "pdf_render_ms": 800.0,
            "line_detection_ms": 1_200.0,
            "vietocr_ms": 27_000.0,
            "tesseract_verifier_ms": 4_100.0,
            "secondary_semantic_verifier_ms": 9_300.0,
            "page_total_ms": 44_400.0,
            "partial_remediation_ms": 2_000.0,
            "partial_remediation": {
                "attempted_count": 2,
                "applied_count": 1,
                "events": [
                    {
                        "crop_id": "p0000-l0009",
                        "applied": True,
                        "reason": "consensus_split_retry_applied",
                        "selected_width": 384,
                        "confidence": 0.91,
                        "elapsed_ms": 1_100.0,
                        "before_text": "Điều 16 có hiệu lực",
                        "after_text": "Điều 15 có hiệu lực",
                    }
                ],
            },
            "tesseract_verifier": {
                "input_line_count": 41,
                "matched_line_count": 38,
                "disagreement_line_count": 24,
            },
            "recognition": {
                "semantic_high_risk_count": 23,
                "semantic_consensus_retry_count": 1,
                "semantic_consensus_retry_attempted_count": 2,
                "semantic_surface_consensus_count": 6,
                "semantic_verifier_consensus_count": 4,
                "semantic_consensus_retry_events": [
                    {
                        "crop_id": "p0000-l0003",
                        "applied": True,
                        "reason": "consensus_split_retry_applied",
                        "selected_width": 320,
                        "confidence": 0.88,
                        "elapsed_ms": 1_250.0,
                        "before_text": "?Điều 5. Nguyên tắc thông ký website thương mại điện tử",
                        "after_text": "Điều 5. Nguyên tắc thông báo đăng ký website thương mại điện tử",
                    },
                    {
                        "crop_id": "p0000-l0008",
                        "applied": False,
                        "reason": "numeric_integrity_failed",
                        "selected_width": None,
                        "confidence": None,
                        "elapsed_ms": 700.0,
                        "before_text": "31 tháng 1 năm 20",
                        "after_text": "31 tháng 1 năm 20",
                    },
                ],
            },
        },
    )

    assert "retry=1/2" in rendered
    assert "render=0.8s" in rendered
    assert "detect=1.2s" in rendered
    assert "VietOCR=27.0s" in rendered
    assert "Tesseract=4.1s" in rendered
    assert "kiểm chứng=9.3s" in rendered
    assert "cứu hộ=2.0s" in rendered
    assert "cứu hộ=1/2" in rendered
    assert "[OCR][Retry] p1:l4 | ÁP DỤNG | width=320 | 1.2s" in rendered
    assert "trước: \"?Điều 5. Nguyên tắc thông ký website thương mại điện tử\"" in rendered
    assert "sau:   \"Điều 5. Nguyên tắc thông báo đăng ký website thương mại điện tử\"" in rendered
    assert "[OCR][Retry] p1:l9 | GIỮ NGUYÊN | width=- | 0.7s" in rendered
    assert "numeric_integrity_failed" in rendered
    assert "[OCR][Rescue] p1:l10 | ÁP DỤNG | width=384 | 1.1s" in rendered
    assert "sau:   \"Điều 15 có hiệu lực\"" in rendered


def test_logging_hides_known_harmless_dependency_warnings_only():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        configure_logging()
        warnings.warn_explicit(
            "No ccache found. Recompiling may be required.",
            UserWarning,
            filename="extension_utils.py",
            lineno=1,
            module="paddle.utils.cpp_extension.extension_utils",
        )
        warnings.warn_explicit(
            "pkg_resources is deprecated as an API.",
            UserWarning,
            filename="gdown/__init__.py",
            lineno=1,
            module="gdown",
        )
        warnings.warn_explicit(
            "OCR model checksum mismatch",
            UserWarning,
            filename="model.py",
            lineno=1,
            module="government_ocr_text_api.model",
        )

    assert [str(item.message) for item in caught] == ["OCR model checksum mismatch"]
