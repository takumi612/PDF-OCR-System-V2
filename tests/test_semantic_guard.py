from government_ocr_text_api.config import Settings
from government_ocr_text_api.semantic_guard import evaluate_semantic_line
from government_ocr_text_api.semantic_retry import (
    RetryVariant,
    choose_consensus_retry,
    choose_verifier_consensus,
    normalize_legal_collocations,
    restore_three_engine_separators,
)


def test_removes_only_unsupported_non_numeric_suffix():
    original = (
        "59/2015/TT-BCT Ngày 31 tháng 12 năm 2015 của Bộ Công "
        "Thương quy định về người thuy nhi viện thuy nghiệp thuyên thiến"
    )
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.47,
        primary_error_code="decoder_loop_trimmed",
        secondary_text=(
            "59/2015/TT-BCT ngày 31 tháng 12 năm 2015 ca B Công "
            "Thưng quy đnh v"
        ),
        secondary_confidence=0.97,
        settings=Settings(),
    )

    assert decision.text.endswith("quy định về")
    assert decision.raw_text == original
    assert decision.reasons == ("unsupported_suffix_removed",)
    assert decision.risk == "medium"


def test_numeric_suffix_is_never_deleted():
    original = "Nội dung điều khoản có hiệu lực năm 2025 2026 2027"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.40,
        primary_error_code="decoder_loop_trimmed",
        secondary_text="Nội dung điều khoản có hiệu lực năm",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.raw_text is None
    assert decision.risk == "high"
    assert "numeric_suffix_protected" in decision.reasons


def test_equal_length_numeric_disagreement_is_masked_without_rewrite():
    original = "Dieu khoan co hieu luc nam 2025"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.95,
        primary_error_code=None,
        secondary_text="Dieu khoan co hieu luc nam 2026",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert decision.reasons == ("secondary_numeric_disagreement",)


def test_short_numeric_suffix_is_protected_and_masked():
    original = "Dieu khoan co hieu luc nam 2025"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.95,
        primary_error_code=None,
        secondary_text="Dieu khoan co hieu luc nam",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert decision.reasons == ("numeric_suffix_protected",)


def test_secondary_omission_disagreement_is_masked_not_rewritten():
    original = "xâm phạm ninh quốc gia, khủng bố"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.85,
        primary_error_code=None,
        secondary_text="xâm phm an ninh quc gia, khng b",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.raw_text is None
    assert decision.risk == "high"
    assert "secondary_indicates_primary_omission" in decision.reasons


def test_secondary_orthographic_loss_does_not_replace_safe_primary():
    original = "ngăn chặn hoạt động của tổ chức, cá nhân ở nước ngoài"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.91,
        primary_error_code=None,
        secondary_text="ngăn chn hot đng ca t chc, cá nhân nưc ngoài",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "none"
    assert decision.reasons == ()


def test_risky_primary_is_high_risk_when_secondary_unavailable():
    decision = evaluate_semantic_line(
        primary_text="Nội dung chưa chắc chắn",
        primary_confidence=0.55,
        primary_error_code="tail_segment_uncertain",
        secondary_text=None,
        secondary_confidence=None,
        settings=Settings(),
    )

    assert decision.text == "Nội dung chưa chắc chắn"
    assert decision.risk == "high"
    assert decision.reasons == ("secondary_unavailable",)


def test_safe_primary_stays_safe_when_secondary_unavailable():
    decision = evaluate_semantic_line(
        primary_text="Nội dung đã rõ",
        primary_confidence=0.95,
        primary_error_code=None,
        secondary_text=None,
        secondary_confidence=None,
        settings=Settings(),
    )

    assert decision.risk == "none"
    assert decision.reasons == ()


def test_fewer_than_three_extra_tokens_are_not_auto_deleted():
    original = "Một nội dung pháp lý đầy đủ có thêm hai từ"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.50,
        primary_error_code="tail_segment_uncertain",
        secondary_text="Một nội dung pháp lý đầy đủ có thêm",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert "unsupported_suffix_removed" not in decision.reasons


def test_short_primary_extra_suffix_is_masked_even_when_primary_is_confident():
    original = "mot hai ba bon nam sau bay rac them"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.95,
        primary_error_code=None,
        secondary_text="mot hai ba bon nam sau bay",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert decision.reasons == ("secondary_suffix_disagreement",)


def test_empty_primary_is_high_risk():
    decision = evaluate_semantic_line(
        primary_text="",
        primary_confidence=0.0,
        primary_error_code="recognition_failed",
        secondary_text="Nội dung secondary",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.risk == "high"
    assert decision.reasons == ("empty_primary_text",)


def test_low_confidence_secondary_cannot_authorize_deletion():
    original = "Một hai ba bốn năm sáu rác thêm ở cuối"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.45,
        primary_error_code="decoder_loop_trimmed",
        secondary_text="Một hai ba bốn năm sáu",
        secondary_confidence=0.70,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert decision.reasons == ("secondary_low_confidence",)


def test_short_secondary_prefix_cannot_authorize_deletion():
    original = "Một hai ba rác thêm ở cuối"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.45,
        primary_error_code="decoder_loop_trimmed",
        secondary_text="Một hai ba",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert "unsupported_suffix_removed" not in decision.reasons


def test_tail_mismatch_cannot_authorize_deletion():
    original = "Một hai ba bốn năm sáu bảy rác thêm ở cuối"
    decision = evaluate_semantic_line(
        primary_text=original,
        primary_confidence=0.45,
        primary_error_code="decoder_loop_trimmed",
        secondary_text="Một hai ba bốn khác hẳn bảy",
        secondary_confidence=0.99,
        settings=Settings(),
    )

    assert decision.text == original
    assert decision.risk == "high"
    assert "unsupported_suffix_removed" not in decision.reasons


def test_accent_insensitive_secondary_does_not_hide_primary_diacritic_risk():
    decision = evaluate_semantic_line(
        primary_text="Cơ quan có thẩm quyên",
        primary_confidence=0.96,
        primary_error_code=None,
        secondary_text="Co quan co tham quyen",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    # The existing accent-poor Paddle secondary remains non-authoritative.
    # A separate accent-aware verifier is responsible for this disagreement.
    assert decision.text == "Cơ quan có thẩm quyên"
    assert decision.risk == "none"


def test_verifier_consensus_repairs_missing_words_in_article_heading():
    decision = choose_verifier_consensus(
        primary_text="?Điều 5. Nguyên tắc thông ký website thương mại điện tử",
        verifier_text=(
            "“Điều 5. Nguyên tắc thông báo, đăng ký website thương mại điện tử"
        ),
        verifier_confidence=0.94,
        secondary_text=(
            "Điu 5. Nguyên tc thông báo, đăng ký website thương mi đin t"
        ),
        secondary_confidence=0.97,
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == (
        "“Điều 5. Nguyên tắc thông báo, đăng ký website thương mại điện tử"
    )
    assert decision.reason == "verifier_secondary_consensus_applied"


def test_verifier_consensus_rejects_unsupported_verifier_prefix_noise():
    primary = "c) Đầu tư, hỗ trợ triển khai"
    decision = choose_verifier_consensus(
        primary_text=primary,
        verifier_text="i c) Đầu tư, hỗ trợ triển khai",
        verifier_confidence=0.95,
        secondary_text="c) Dau tu, ho tro trien khai",
        secondary_confidence=0.97,
        settings=Settings(),
    )

    assert decision.applied is False
    assert decision.text == primary
    assert decision.reason == "verifier_consensus_unsupported_tokens"


def test_verifier_consensus_prefers_high_confidence_accent_aware_wording():
    decision = choose_verifier_consensus(
        primary_text="Luật Phòng, chống rủa tiền",
        verifier_text="Luật Phòng, chống rửa tiền",
        verifier_confidence=0.97,
        secondary_text="Luat Phong, chong rua tien",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == "Luật Phòng, chống rửa tiền"
    assert decision.reason == "verifier_diacritic_consensus_applied"


def test_verifier_consensus_keeps_primary_diacritics_below_strict_threshold():
    primary = "Luật Phòng, chống rủa tiền"
    decision = choose_verifier_consensus(
        primary_text=primary,
        verifier_text="Luật Phòng, chống rửa tiền",
        verifier_confidence=0.90,
        secondary_text="Luat Phong, chong rua tien",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.applied is False
    assert decision.text == primary


def test_verifier_word_insertion_preserves_aligned_primary_diacritics():
    decision = choose_verifier_consensus(
        primary_text="chặn hoạt động của tổ chức tài trợ tiền",
        verifier_text="chặn hoạt động của tô chức và cá nhân tài trợ tiên",
        verifier_confidence=0.97,
        secondary_text="chan hoat dong cua to chuc va ca nhan tai tro tien",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == "chặn hoạt động của tổ chức và cá nhân tài trợ tiền"


def test_verifier_does_not_swap_equal_strength_tone_without_lexical_evidence():
    primary = "Trao đồi, cung cấp thông tin"
    decision = choose_verifier_consensus(
        primary_text=primary,
        verifier_text="Trao đổi, cung cấp thông tin",
        verifier_confidence=0.98,
        secondary_text="Trao doi, cung cap thong tin",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.applied is False
    assert decision.text == primary


def test_verifier_applies_safe_token_upgrade_without_degrading_neighbor():
    decision = choose_verifier_consensus(
        primary_text="phương; đúng chức năng, nhiệm vu, quyền hạn",
        verifier_text="phương; đúng chức năng, nhiệm vụ, quyên hạn",
        verifier_confidence=0.98,
        secondary_text="phuong dung chuc nang nhiem vu quyen han",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == "phương; đúng chức năng, nhiệm vụ, quyền hạn"


def test_verifier_adds_missing_circumflex_without_trading_tone():
    decision = choose_verifier_consensus(
        primary_text="tài sản cho đói tượng",
        verifier_text="tài sản cho đối tượng",
        verifier_confidence=0.96,
        secondary_text="tai san cho doi tuong",
        secondary_confidence=0.98,
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == "tài sản cho đối tượng"


def test_consensus_retry_restores_numeric_legal_content_and_tesseract_diacritics():
    decision = choose_consensus_retry(
        primary_text="cho nguy nguy nguy nhi phương có tỷ lệ điều tiết",
        verifier_text=(
            "c) 50% nhu cầu kinh phí tăng thêm cho các địa phương "
            "có tỷ lệ điều tiết"
        ),
        verifier_confidence=0.96,
        secondary_text=(
            "c) 50% nhu cu kinh phí tăng thêm cho các đa phương có t l điu tit"
        ),
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "c) 50% nhu cầu kinh phi tăng thêm cho các địa phương "
                "có tỷ lệ điều tiết",
                0.88,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == (
        "c) 50% nhu cầu kinh phí tăng thêm cho các địa phương có tỷ lệ điều tiết"
    )
    assert decision.reason == "consensus_split_retry_applied"


def test_consensus_retry_fuses_exact_primary_and_verifier_token_agreement():
    decision = choose_consensus_retry(
        primary_text=(
            "Ca Ch Tiến nghị, đề xuất việc sửa đổi, bổ sung và biện pháp "
            "xử lý tháo gỡ"
        ),
        verifier_text=(
            ". ©) Kién nghi, đề xuất việc sửa đổi, bd sung và biện pháp "
            "xử lý tháo gỡ"
        ),
        verifier_confidence=0.90,
        secondary_text=(
            "c) Kin nghi, đ xut vic sa đi, b sung và bin pháp x lý tháo g"
        ),
        secondary_confidence=0.96,
        variants=(
            RetryVariant(
                "c) Kiến nghị, để xuất việc sửa đổi, bồ sung và biện pháp "
                "xử lý tháo gỡ",
                0.85,
                288,
            ),
            RetryVariant(
                "c) Kiến nghị, để xuất việc sửa đổi, bổ sung và biện pháp "
                "xử lý tháo gỡ",
                0.84,
                320,
            ),
            RetryVariant(
                "c) Kiến nghị, đề xuất việc sửa đồi, bồ sung và biện pháp "
                "xứ lý tháo gỡ",
                0.83,
                352,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.selected_width == 320
    assert decision.text == (
        "c) Kiến nghị, đề xuất việc sửa đổi, bổ sung và biện pháp xử lý tháo gỡ"
    )


def test_consensus_retry_rejects_any_numeric_disagreement():
    decision = choose_consensus_retry(
        primary_text="Điều 14 có hiệu lực",
        verifier_text="Điều 44 có hiệu lực",
        verifier_confidence=0.97,
        secondary_text="Điều 24 có hiệu lực",
        secondary_confidence=0.98,
        variants=(RetryVariant("Điều 44 có hiệu lực", 0.91, 320),),
        settings=Settings(),
    )

    assert decision.applied is False
    assert decision.text == "Điều 14 có hiệu lực"
    assert decision.reason == "consensus_retry_numeric_disagreement"


def test_consensus_retry_rejects_low_confidence_verifier():
    decision = choose_consensus_retry(
        primary_text="Nội dung sai",
        verifier_text="Nội dung đúng",
        verifier_confidence=0.60,
        secondary_text="Noi dung dung",
        secondary_confidence=0.98,
        variants=(RetryVariant("Nội dung đúng", 0.90, 320),),
        settings=Settings(),
    )

    assert decision.applied is False
    assert decision.text == "Nội dung sai"
    assert decision.reason == "consensus_retry_insufficient_evidence"


def test_consensus_retry_does_not_rewrite_diacritic_only_primary_line():
    primary = "Thương quy định về quản lý hoạt động thương mại điện tử"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text="Thương quy định về quan lý hoạt động thương mại điện tử",
        verifier_confidence=0.96,
        secondary_text="Thuong quy đinh ve quan ly hoat đong thuong mai đien tu",
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "Thương quy định về quan lý hoạt động thương mại điện tử",
                0.86,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is False
    assert decision.text == primary
    assert decision.reason == "consensus_retry_no_material_gain"


def test_numeric_retry_changes_only_number_span_and_preserves_primary_words():
    primary = "2. Sửa đổi, bổ sung điểm c khoản 2 Điều 1 7 như sau:"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text="2. Sửa đồi, bồ sung điểm c khoản 2 Điều 17 như sau:",
        verifier_confidence=0.94,
        secondary_text="2. Sua đoi, bo sung điem c khoan 2 Đieu 17 nhu sau:",
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "2. Sửa đồi, bồ sung điểm c khoản 2 Điều 17 như sau:",
                0.86,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == "2. Sửa đổi, bổ sung điểm c khoản 2 Điều 17 như sau:"


def test_omission_retry_keeps_aligned_primary_diacritics():
    primary = "điện tử đã hoàn thành tục đăng ký và xác nhận của Bộ Công"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text=(
            "điện tử đã hoàn thành thủ tục đăng ký và xác nhận của Bộ Cong’"
        ),
        verifier_confidence=0.93,
        secondary_text=(
            "đien tu đa hoan thanh thu tuc đang ky va xac nhan cua Bo Cong"
        ),
        secondary_confidence=0.96,
        variants=(
            RetryVariant(
                "điện tử đã hoàn thành thủ tục đăng ký và xác nhận của Bộ Cong",
                0.84,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == (
        "điện tử đã hoàn thành thủ tục đăng ký và xác nhận của Bộ Công"
    )


def test_consensus_retry_restores_verifier_punctuation_when_tokens_match():
    primary = (
        "tài sản công, Luật Quản lý thuế, Luật Thuế thu nhập cá nhân, "
        "Lự trữ quốc"
    )
    expected = (
        "tài sản công, Luật Quản lý thuế, Luật Thuế thu nhập cá nhân, "
        "Luật Dự trữ quốc"
    )
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text=expected,
        verifier_confidence=0.96,
        secondary_text=(
            "tai san cong, Luat Quan ly thue, Luat Thue thu nhap ca nhan, "
            "Luat Du tru quoc"
        ),
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "tài sản công, Luật Quản lý thuế Luật Thuế thu nhập cá nhân "
                "Luật Dự trữ quốc",
                0.89,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == expected


def test_consensus_retry_uses_candidate_and_verifier_capitalization():
    primary = (
        "g) biện pháp xử lý vi phạm đối những người không tuân thủ quy chế"
    )
    expected = (
        "g) Biện pháp xử lý vi phạm đối với những người không tuân thủ quy chế"
    )
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text=expected,
        verifier_confidence=0.96,
        secondary_text=(
            "g) Bien phap xu ly vi pham doi voi nhung nguoi khong tuan thu quy che"
        ),
        secondary_confidence=0.97,
        variants=(RetryVariant(expected, 0.87, 320),),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == expected


def test_consensus_retry_restores_trailing_legal_punctuation():
    primary = "2. Sửa đổi, bổ sung điểm c khoản 2 Điều 1 7 như sau:"
    expected = "2. Sửa đổi, bổ sung điểm c khoản 2 Điều 17 như sau:"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text=expected,
        verifier_confidence=0.96,
        secondary_text="2. Sua doi, bo sung diem c khoan 2 Dieu 17 nhu sau:",
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "2. Sửa đổi, bổ sung điểm c khoản 2 Điều 17 như sau",
                0.86,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == expected


def test_consensus_retry_does_not_corrupt_correct_unspaced_list_item():
    primary = "4.Bổ sung Điều 20a dưới Điều 20 tại Chương II như sau:"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text="4. Bố sung Điều 20a dưới Điều 20 tại Chương II như sau:",
        verifier_confidence=0.96,
        secondary_text="4. B sung Dieu 20a duoi Dieu 20 tai Chuong II nhu sau:",
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "4 Bồ sung Điều 20a dưới Điều 20 tại Chương II như sau:",
                0.87,
                384,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == "4. Bổ sung Điều 20a dưới Điều 20 tại Chương II như sau:"
    assert decision.reason == "consensus_surface_repair_applied"


def test_surface_repair_cleans_noise_before_verified_opening_quote():
    primary = "- Chương II. Thủ tục thông báo, đăng ký website thương mại điện tử và"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text=". “Chương II. Thủ tục thông báo, đăng ký website thương mại điện tử và",
        verifier_confidence=0.96,
        secondary_text="Chuong II. Thu tuc thong bao, dang ky website thuong mai dien tu va",
        secondary_confidence=0.97,
        variants=(),
        settings=Settings(),
    )

    assert decision.text == (
        "“Chương II. Thủ tục thông báo, đăng ký website thương mại điện tử và"
    )
    assert decision.applied is True


def test_surface_repair_rejects_repeated_terminal_punctuation():
    primary = "- Văn phòng Trung ương và các Ban của Đảng"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text="- Văn phòng Trung ương và các Ban của Đảng; ;",
        verifier_confidence=0.96,
        secondary_text="Van phong Trung uong va cac Ban cua Dang",
        secondary_confidence=0.97,
        variants=(),
        settings=Settings(),
    )

    assert decision.text == "- Văn phòng Trung ương và các Ban của Đảng;"
    assert decision.applied is True


def test_surface_repair_does_not_replace_existing_terminal_punctuation_kind():
    primary = "- Văn phòng Trung ương Đảng và các Ban của Đảng;"
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text="- Văn phòng Trung ương Đảng và các Ban của Đảng:",
        verifier_confidence=0.96,
        secondary_text="Van phong Trung uong Dang va cac Ban cua Dang",
        secondary_confidence=0.97,
        variants=(),
        settings=Settings(),
    )

    assert decision.text == primary
    assert decision.applied is False


def test_surface_repair_prefers_verified_unicode_opening_quote():
    primary = '?a) Cơ sở công nghiệp quốc phòng nòng cốt trích lập Quỹ'
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text='“a) Cơ sở công nghiệp quốc phòng nòng cốt trích lập Quỹ',
        verifier_confidence=0.96,
        secondary_text='a) Co so cong nghiep quoc phong nong cot trich lap Quy',
        secondary_confidence=0.97,
        variants=(),
        settings=Settings(),
    )

    assert decision.text == '“a) Cơ sở công nghiệp quốc phòng nòng cốt trích lập Quỹ'
    assert decision.applied is True


def test_surface_repair_upgrades_straight_quote_to_verified_unicode_quote():
    primary = '"c) Số định danh cá nhân đối với khách hàng cá nhân'
    decision = choose_consensus_retry(
        primary_text=primary,
        verifier_text='“c) Số định danh cá nhân đối với khách hàng cá nhân',
        verifier_confidence=0.96,
        secondary_text='c) So dinh danh ca nhan doi voi khach hang ca nhan',
        secondary_confidence=0.97,
        variants=(),
        settings=Settings(),
    )

    assert decision.text == '“c) Số định danh cá nhân đối với khách hàng cá nhân'
    assert decision.applied is True


def test_three_engine_separator_consensus_overrides_primary_comma():
    primary = "Nam, số, ngày cấp giấy chứng nhận đăng ký doanh nghiệp"
    verifier = "Nam; sô, ngày cap Giây chứng nhận đăng ký doanh nghiệp"
    secondary = "Nam; s, ngày cp Giy chúng nhn đăng ký doanh nghip"

    assert restore_three_engine_separators(primary, verifier, secondary) == (
        "Nam; số, ngày cấp giấy chứng nhận đăng ký doanh nghiệp"
    )


def test_three_engine_separator_consensus_does_not_override_without_agreement():
    primary = "quốc phòng; quy định việc thành lập"
    verifier = "quốc phòng: quy định việc thành lập"
    secondary = "quốc phòng; quy định việc thành lập"

    assert restore_three_engine_separators(primary, verifier, secondary) == primary


def test_normalizes_only_unambiguous_legal_context_diacritics():
    value = (
        "các tinh, thành phố trực thuộc Trung ương; mở tài khoản tai thành viên; "
        "giai đoan 2023 - 2025; chấp nhận rủ ro; cơ cở pháp lý; "
        "số lượng hợp đồng năm giữ; Uu tiên phân bự nguồn lực; "
        "sự tham gia của các tố; các tinh, thành phố trực thuộc"
    )

    assert normalize_legal_collocations(value) == (
        "các tỉnh, thành phố trực thuộc Trung ương; mở tài khoản tại thành viên; "
        "giai đoạn 2023 - 2025; chấp nhận rủi ro; cơ sở pháp lý; "
        "số lượng hợp đồng nắm giữ; Ưu tiên phân bổ nguồn lực; "
        "sự tham gia của các tổ; các tỉnh, thành phố trực thuộc"
    )


def test_consensus_retry_normalizes_unambiguous_legal_collocation():
    decision = choose_consensus_retry(
        primary_text=(
            "2.00 Sở Xác định địa phương nhận bồ sung cân đối ngân sách từ"
        ),
        verifier_text=(
            "2. Cơ sở xác định địa phương nhận bd sung cân đối ngân sách từ"
        ),
        verifier_confidence=0.96,
        secondary_text=(
            "2. Co so xac dinh dia phuong nhan b sung can doi ngan sach tu"
        ),
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "2. Cơ sở xác định địa phương nhận bồ sung cân đối ngân sách từ",
                0.85,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == (
        "2. Cơ sở xác định địa phương nhận bổ sung cân đối ngân sách từ"
    )


def test_consensus_retry_keeps_terminal_punctuation_confirmed_by_two_engines():
    expected = (
        "c) Đầu tư, hỗ trợ triển khai thực hiện các chương trình, dự án, hoạt động,"
    )
    decision = choose_consensus_retry(
        primary_text=(
            "Ca ch cầ tượ trợ trợển khai thực hiện các chương trình, dự án, hoạt động,"
        ),
        verifier_text=(
            "i c) Đầu tư, hỗ trợ triển khai thực hiện các chương trình, dự án, hoạt động,"
        ),
        verifier_confidence=0.95,
        secondary_text=(
            "c) Dau tu, ho tro trien khai thuc hien cac chuong trinh, du an, hoat dong,"
        ),
        secondary_confidence=0.97,
        variants=(
            RetryVariant(
                "c) Đầu tư, hỗ trợ triển khai thực hiện các chương trình, dự án, hoạt động",
                0.88,
                320,
            ),
        ),
        settings=Settings(),
    )

    assert decision.applied is True
    assert decision.text == expected
