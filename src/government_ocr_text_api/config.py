from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TensorCleanupPolicy(str, Enum):
    NEVER = "never"
    DOCUMENT_END = "document_end"
    EVERY_N_PAGES = "every_n_pages"
    ON_MEMORY_PRESSURE = "on_memory_pressure"


class NativeFailurePolicy(str, Enum):
    PDFIUM_THEN_OCR = "pdfium_then_ocr"
    OCR_ALL = "ocr_all"
    FAIL = "fail"


class Settings(BaseSettings):
    """Cấu hình tập trung; giữ tên và giá trị mặc định từ dự án OCR gốc."""

    model_config = SettingsConfigDict(
        env_prefix="GOVERNMENT_OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Government OCR Text API"
    app_version: str = "1.3.5.9"

    # Cấu hình OCR kế thừa.
    device: str = "cpu"
    fallback_render_dpi: int = Field(default=200, ge=72, le=600)
    full_page_image_min_coverage: float = Field(default=0.98, ge=0.0, le=1.0)
    max_skew_without_correction: float = Field(default=1.0, ge=0.0, le=10.0)
    line_crop_padding_ratio: float = Field(default=0.04, ge=0.0, le=0.5)
    low_ink_ratio: float = Field(default=0.0005, ge=0.0, le=1.0)

    vietocr_config_name: str = "vgg_seq2seq"
    vietocr_config_path: Path = Path("models/vietocr/vgg_seq2seq.yml")
    vietocr_weights_path: Path = Path("models/vietocr/vgg_seq2seq.pth")
    vietocr_beamsearch: bool = False
    vietocr_image_height: int = Field(default=32, ge=16, le=128)
    line_detection_model_name: str = "PP-OCRv6_medium_det"
    layout_model_name: str = "PP-DocLayoutV3"  # legacy compatibility; không chạy trong text-only API
    review_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    recognition_batch_size: int = Field(default=32, ge=1, le=32)
    # Tối ưu OCR scan: nhận dạng nhiều trang theo cửa sổ và gom batch theo
    # resized-width/pixel budget thay vì 5 bucket cố định.
    recognition_window_pages: int = Field(default=3, ge=1, le=8)
    recognition_pixel_budget: int = Field(default=12288, ge=512, le=65536)
    recognition_batch_width_ratio: float = Field(default=2.0, ge=1.0, le=4.0)
    pad_batches_to_common_width: bool = False
    split_wide_crops: bool = True
    # Luôn thử chia an toàn khi crop sẽ bị VietOCR width-cap. Field threshold
    # cũ vẫn được giữ để tương thích .env, nhưng chỉ dùng khi option này tắt.
    split_width_capped_crops: bool = True
    wide_crop_split_threshold_ratio: float = Field(default=1.05, ge=1.0, le=6.0)
    wide_crop_target_ratio: float = Field(default=0.92, ge=0.5, le=1.0)
    wide_crop_max_segments: int = Field(default=6, ge=2, le=12)
    wide_crop_valley_max_ink_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    # Overlap ảnh là thử nghiệm: trên dữ liệu thật, khi hai segment nhận dạng
    # khác nhau ở vùng chung thì seam merge không thể khử trùng an toàn. Mặc
    # định 1.3.1 tắt overlap để tránh chèn hàng loạt token vào nội dung.
    experimental_overlap_split: bool = False
    wide_crop_overlap_height_ratio: float = Field(default=0.0, ge=0.0, le=2.0)
    seam_max_token_overlap: int = Field(default=4, ge=1, le=12)
    seam_max_char_overlap: int = Field(default=32, ge=4, le=128)
    tail_segment_validation: bool = True
    # Retry tail từng làm thay đổi output nhưng chưa chứng minh cải thiện trên
    # corpus pháp lý; chỉ bật khi benchmark A/B có ground truth.
    tail_segment_retry_enabled: bool = False
    tail_segment_min_confidence: float = Field(default=0.58, ge=0.0, le=1.0)
    tail_segment_low_ink_ratio: float = Field(default=0.012, ge=0.0, le=1.0)
    tail_segment_trailing_blank_ratio: float = Field(default=0.18, ge=0.0, le=0.95)
    tail_segment_max_retries_per_page: int = Field(default=2, ge=0, le=16)
    suppress_low_ink_tail_hallucinations: bool = True

    # Multi-view hallucination guard. Không dùng danh sách từ cấm hoặc LLM; chỉ
    # xóa token khi pass trên crop sát mực là một subsequence của output gốc và
    # hình học biên ảnh xác nhận phần bị xóa không có đủ bằng chứng pixel.
    hallucination_guard_enabled: bool = False
    hallucination_guard_apply_changes: bool = True
    hallucination_guard_midline_enabled: bool = False
    hallucination_guard_suffix_only: bool = True
    hallucination_guard_numeric_apply_changes: bool = False
    hallucination_guard_max_rechecks_per_page: int = Field(default=4, ge=0, le=32)
    hallucination_guard_risk_threshold: float = Field(default=0.48, ge=0.0, le=1.0)
    hallucination_guard_min_confidence: float = Field(default=0.78, ge=0.0, le=1.0)
    hallucination_guard_confidence_tolerance: float = Field(default=0.04, ge=0.0, le=0.5)
    hallucination_guard_midline_confidence_bonus: float = Field(default=0.01, ge=0.0, le=0.5)
    hallucination_guard_low_ink_ratio: float = Field(default=0.018, ge=0.0, le=1.0)
    hallucination_guard_edge_blank_ratio: float = Field(default=0.10, ge=0.0, le=0.95)
    hallucination_guard_min_crop_change_ratio: float = Field(default=0.06, ge=0.0, le=0.95)
    hallucination_guard_numeric_min_crop_change_ratio: float = Field(default=0.16, ge=0.0, le=0.95)
    hallucination_guard_margin_height_ratio: float = Field(default=0.12, ge=0.0, le=1.0)
    hallucination_guard_detached_gap_height_ratio: float = Field(default=0.75, ge=0.2, le=4.0)
    hallucination_guard_max_removed_ink_ratio: float = Field(default=0.28, ge=0.0, le=0.8)
    hallucination_guard_max_removed_token_ratio: float = Field(default=0.35, ge=0.01, le=0.8)
    hallucination_guard_min_anchor_tokens: int = Field(default=1, ge=1, le=8)
    hallucination_guard_chars_per_ink_column: float = Field(default=0.20, ge=0.01, le=2.0)
    hallucination_guard_log_event_limit: int = Field(default=6, ge=0, le=50)

    # Decoder-evidence guard cho VietOCR seq2seq. Cơ chế đọc probability và
    # attention trực tiếp từ decoder, ánh xạ attention vào mật độ mực của ảnh,
    # rồi chỉ cắt suffix khi nhiều tín hiệu độc lập cùng xác nhận decoder đã
    # dừng tiến trên ảnh hoặc tiếp tục sinh sau khi bằng chứng ảnh đã hết.
    decoder_evidence_enabled: bool = True
    decoder_evidence_apply_changes: bool = True
    decoder_evidence_max_checks_per_page: int = Field(default=8, ge=0, le=10000)
    decoder_evidence_trace_batch_size: int = Field(default=12, ge=1, le=64)
    decoder_evidence_candidate_score_threshold: float = Field(default=0.38, ge=0.0, le=1.0)
    decoder_evidence_max_decode_steps: int = Field(default=128, ge=16, le=512)
    decoder_evidence_candidate_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    decoder_evidence_min_prefix_chars: int = Field(default=8, ge=1, le=128)
    decoder_evidence_window_tokens: int = Field(default=6, ge=3, le=24)
    decoder_evidence_min_unsupported_tokens: int = Field(default=3, ge=2, le=16)
    decoder_evidence_low_token_probability: float = Field(default=0.56, ge=0.0, le=1.0)
    decoder_evidence_low_ink_support: float = Field(default=0.10, ge=0.0, le=1.0)
    decoder_evidence_relative_probability_ratio: float = Field(default=0.68, ge=0.1, le=1.0)
    decoder_evidence_relative_ink_ratio: float = Field(default=0.28, ge=0.05, le=1.0)
    decoder_evidence_probability_threshold_cap: float = Field(default=0.72, ge=0.0, le=1.0)
    decoder_evidence_ink_threshold_cap: float = Field(default=0.18, ge=0.0, le=1.0)
    decoder_evidence_stall_range_ratio: float = Field(default=0.035, ge=0.001, le=0.5)
    decoder_evidence_end_margin_ratio: float = Field(default=0.06, ge=0.0, le=0.5)
    decoder_evidence_near_loop_similarity: float = Field(default=0.74, ge=0.5, le=1.0)
    decoder_evidence_near_loop_min_tokens: int = Field(default=8, ge=4, le=40)
    decoder_evidence_numeric_apply_changes: bool = False
    decoder_evidence_cluster_word_expansion_enabled: bool = True
    decoder_evidence_cluster_max_words: int = Field(default=2, ge=0, le=4)
    decoder_evidence_cluster_attention_range_ratio: float = Field(default=0.12, ge=0.005, le=0.5)
    decoder_evidence_cluster_center_distance_ratio: float = Field(default=0.08, ge=0.005, le=0.5)
    decoder_evidence_cluster_end_margin_ratio: float = Field(default=0.10, ge=0.0, le=0.5)
    decoder_evidence_cluster_support_ratio: float = Field(default=0.35, ge=0.05, le=1.0)
    decoder_evidence_cluster_support_cap: float = Field(default=0.16, ge=0.0, le=1.0)
    decoder_evidence_cluster_probability_ratio: float = Field(default=0.90, ge=0.1, le=1.0)
    decoder_evidence_cluster_probability_bonus: float = Field(default=0.12, ge=0.0, le=0.5)
    decoder_evidence_cluster_probability_cap: float = Field(default=0.82, ge=0.0, le=1.0)
    decoder_evidence_midline_enabled: bool = True
    decoder_evidence_full_coverage: bool = False
    # Khi selective evidence chọn một segment của dòng đã split, trace toàn bộ
    # segment cùng line để giữ đủ right/left anchor cho cross-segment validation.
    decoder_evidence_include_split_line_context: bool = True
    decoder_evidence_midline_anchor_words: int = Field(default=2, ge=1, le=6)
    decoder_evidence_midline_min_words: int = Field(default=1, ge=1, le=8)
    decoder_evidence_midline_max_words: int = Field(default=12, ge=1, le=40)
    decoder_evidence_midline_min_suffix_chars: int = Field(default=6, ge=1, le=128)
    decoder_evidence_midline_max_span_ratio: float = Field(default=0.55, ge=0.05, le=0.9)
    decoder_evidence_midline_probability_bonus: float = Field(default=0.10, ge=0.0, le=0.5)
    decoder_evidence_midline_support_multiplier: float = Field(default=1.25, ge=1.0, le=4.0)
    decoder_evidence_midline_anchor_probability_ratio: float = Field(default=1.20, ge=1.0, le=4.0)
    decoder_evidence_midline_anchor_support_ratio: float = Field(default=1.80, ge=1.0, le=8.0)
    decoder_evidence_midline_relative_probability_ratio: float = Field(default=0.82, ge=0.1, le=1.0)
    decoder_evidence_midline_relative_support_ratio: float = Field(default=0.55, ge=0.05, le=1.0)
    decoder_evidence_midline_attention_range_ratio: float = Field(default=0.12, ge=0.005, le=0.6)
    decoder_evidence_midline_attention_progress_ratio: float = Field(default=0.04, ge=0.0, le=0.5)
    decoder_evidence_midline_recurrence_distance_ratio: float = Field(default=0.10, ge=0.005, le=0.5)
    decoder_evidence_midline_recovery_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    decoder_evidence_midline_anchor_progress_ratio: float = Field(default=0.08, ge=0.0, le=0.8)
    decoder_evidence_event_limit: int = Field(default=50, ge=0, le=500)

    # Line-level visual grounding. Attention is mapped from each split segment
    # back to the original line coordinate. 1.3.5.5 uses continuous attention
    # overlap instead of an absolute per-bin threshold so the signal is stable
    # for diffuse real-world attention distributions.
    decoder_evidence_visual_grounding_enabled: bool = True
    decoder_evidence_cross_segment_enabled: bool = True
    decoder_evidence_cross_segment_anchor_words: int = Field(default=2, ge=1, le=6)
    decoder_evidence_cross_segment_max_words: int = Field(default=12, ge=1, le=40)
    decoder_evidence_cross_segment_min_anchor_probability: float = Field(default=0.58, ge=0.0, le=1.0)
    decoder_evidence_cross_segment_min_anchor_support: float = Field(default=0.10, ge=0.0, le=1.0)
    decoder_evidence_cross_segment_relative_support_ratio: float = Field(default=0.68, ge=0.05, le=1.0)
    decoder_evidence_cross_segment_relative_coverage_ratio: float = Field(default=0.58, ge=0.05, le=1.0)
    decoder_evidence_cross_segment_coverage_gain_cap: float = Field(default=0.20, ge=0.0, le=1.0)
    decoder_evidence_cross_segment_relative_probability_ratio: float = Field(default=0.92, ge=0.1, le=1.0)
    decoder_evidence_cross_segment_strong_visual_override_support: float = Field(default=0.055, ge=0.0, le=1.0)
    decoder_evidence_cross_segment_min_reuse_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    decoder_evidence_cross_segment_max_progress_ratio: float = Field(default=0.045, ge=0.0, le=0.5)
    decoder_evidence_cross_segment_global_backtrack_tolerance: float = Field(default=0.03, ge=0.0, le=0.5)
    decoder_evidence_cross_segment_rejection_log_limit: int = Field(default=12, ge=0, le=100)

    # Independent visual verifier enabled by default for semantic comparison.
    secondary_recognizer_enabled: bool = True
    secondary_recognizer_model_name: str = "latin_PP-OCRv5_mobile_rec"
    secondary_recognizer_min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    secondary_recognizer_preference_margin: float = Field(default=0.08, ge=0.0, le=0.5)
    secondary_recognizer_apply_changes: bool = False

    # Semantic-safe verifier. The secondary recognizer never supplies returned
    # characters; it may only validate a deletion-only suffix proposal or mark
    # the primary line unsafe for downstream AI consumption.
    semantic_verification_enabled: bool = True
    # Run PP-OCRv5 only after the page-wide Tesseract pass and only for lines
    # carrying primary/Tesseract risk. If Tesseract is disabled, the pipeline
    # conservatively falls back to checking every line.
    semantic_selective_verification_enabled: bool = True
    semantic_auto_trim_enabled: bool = True
    semantic_secondary_min_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    semantic_primary_low_confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    semantic_suffix_min_extra_tokens: int = Field(default=3, ge=2, le=16)
    semantic_prefix_min_tokens: int = Field(default=6, ge=2, le=64)
    semantic_position_match_ratio: float = Field(default=0.72, ge=0.0, le=1.0)
    semantic_tail_match_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    semantic_material_similarity: float = Field(default=0.72, ge=0.0, le=1.0)
    semantic_retry_enabled: bool = True
    semantic_retry_min_verifier_confidence: float = Field(default=0.88, ge=0.0, le=1.0)
    semantic_retry_min_diacritic_verifier_confidence: float = Field(
        default=0.94, ge=0.0, le=1.0
    )
    semantic_retry_min_secondary_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    semantic_retry_min_variant_confidence: float = Field(default=0.78, ge=0.0, le=1.0)
    semantic_retry_min_verifier_similarity: float = Field(default=0.88, ge=0.0, le=1.0)
    semantic_retry_min_secondary_similarity: float = Field(default=0.72, ge=0.0, le=1.0)
    semantic_retry_min_material_gain: float = Field(default=0.015, ge=0.0, le=0.5)
    semantic_retry_widths: str = "288,320,384"
    semantic_retry_crop_padding_height_ratio: float = Field(default=0.12, ge=0.0, le=1.0)
    semantic_retry_primary_max_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    semantic_retry_max_lines_per_page: int = Field(default=32, ge=0, le=32)
    # Một lượt cứu hộ cuối cho các dòng vẫn high-risk sau pipeline chính.
    # Tesseract chạy trên crop riêng với PSM 7; chỉ đồng thuận ba engine mới
    # được phép thay text, vì vậy cơ chế này không biến PARTIAL thành kết quả
    # "đẹp giả" bằng suy đoán từ điển.
    partial_remediation_enabled: bool = True
    partial_remediation_max_lines_per_page: int = Field(default=8, ge=0, le=32)
    partial_remediation_tesseract_psm: int = Field(default=7, ge=1, le=13)

    # Independent, accent-aware page verifier. It never supplies replacement
    # characters: a disagreement can only mark a line for human review.
    tesseract_verifier_enabled: bool = True
    tesseract_executable_path: Path = Path("tools/tesseract/tesseract.exe")
    tesseract_data_path: Path = Path("tools/tesseract/tessdata_best")
    tesseract_languages: str = "vie+eng"
    tesseract_page_segmentation_mode: int = Field(default=3, ge=1, le=13)
    tesseract_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    tesseract_fail_closed: bool = True
    tesseract_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    tesseract_line_match_score: float = Field(default=0.55, ge=0.0, le=1.0)
    tesseract_material_similarity: float = Field(default=0.55, ge=0.0, le=1.0)

    # Low-cost profiling is intentionally enabled during the semantic-stability
    # phase. It does not optimize runtime; it prevents performance regressions
    # from becoming invisible technical debt.
    performance_diagnostics_enabled: bool = True

    # CPU runtime tuning 1.3.5.6. Torch và Paddle được cấu hình riêng để tránh
    # tình trạng runtime vô tình khóa inference ở một thread. Paddle oneDNN/MKL-DNN
    # có fallback an toàn nếu backend hiện tại không khởi tạo được.
    cpu_runtime_tuning_enabled: bool = True
    torch_cpu_threads: int = Field(default=4, ge=1, le=64)
    torch_interop_threads: int = Field(default=1, ge=1, le=16)
    paddle_cpu_threads: int = Field(default=4, ge=1, le=64)
    paddle_enable_mkldnn: bool = False
    paddle_mkldnn_fallback: bool = True

    # Conservative pre-recognition filter cho crop gần như blank hoặc chỉ là
    # horizontal rule/dotted leader. Không đọc lexical content và không áp dụng
    # cho crop có vertical ink span giống glyph thật.
    ocr_nontext_crop_filter_enabled: bool = True
    ocr_nontext_min_ink_ratio: float = Field(default=0.0015, ge=0.0, le=0.1)
    ocr_nontext_rule_max_ink_ratio: float = Field(default=0.08, ge=0.0, le=0.5)
    ocr_nontext_rule_max_active_row_ratio: float = Field(default=0.15, ge=0.01, le=1.0)
    ocr_nontext_rule_min_ink_column_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    ocr_nontext_rule_min_aspect_ratio: float = Field(default=4.0, ge=1.0, le=50.0)
    ocr_nontext_rule_max_component_height_ratio: float = Field(default=0.20, ge=0.05, le=1.0)
    ocr_nontext_rule_min_small_component_ratio: float = Field(default=0.80, ge=0.5, le=1.0)
    ocr_nontext_rule_min_component_count: int = Field(default=5, ge=2, le=100)

    ocr_column_aware_ordering: bool = True
    ocr_column_occupancy_threshold: float = Field(default=0.14, ge=0.0, le=1.0)
    ocr_column_min_gap_ratio: float = Field(default=0.03, ge=0.005, le=0.25)
    ocr_signature_block_enabled: bool = True
    ocr_signature_block_max_right_lines: int = Field(default=6, ge=2, le=12)
    merge_baseline_fragments: bool = True
    baseline_fragment_max_gap_ratio: float = Field(default=1.0, ge=0.1, le=4.0)
    baseline_fragment_narrow_width_ratio: float = Field(default=1.8, ge=0.5, le=6.0)

    # Retry cực hẹp cho tiêu đề chương bị detector bỏ sót ký tự La Mã nằm
    # sát bên phải. Chỉ chấp nhận output dạng ``Chương <số La Mã/Ả Rập>``.
    chapter_heading_retry_enabled: bool = True
    # Comma-separated labels; escaped before regex compilation. This keeps the
    # structural retry configurable instead of coupling it to one corpus word.
    chapter_heading_retry_labels: str = "chương"
    chapter_heading_expand_height_ratio: float = Field(default=4.0, ge=1.0, le=10.0)
    chapter_heading_min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    chapter_heading_max_retries_per_page: int = Field(default=2, ge=0, le=8)

    beam_retry_enabled: bool = True
    max_beam_retries_per_page: int = Field(default=1, ge=0, le=16)
    use_torch_inference_mode: bool = True

    max_pdf_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1024)
    max_pages: int = Field(default=5000, ge=1)
    web_upload_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1024)
    web_max_concurrent_requests: int = Field(default=1, ge=1, le=4)
    # Các field legacy giữ để tái sử dụng .env của dự án cũ.
    web_runtime_root: Path = Path("runtime/jobs")
    web_max_pending_jobs: int = Field(default=1, ge=1)
    web_poll_interval_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    web_ignored_overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    web_low_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    runner_mode: str = "inprocess"
    request_temp_root: Path = Path("runtime/tmp")

    tensor_cleanup_policy: TensorCleanupPolicy = TensorCleanupPolicy.DOCUMENT_END
    tensor_cleanup_every_pages: int = Field(default=50, ge=1)
    memory_pressure_percent: float = Field(default=85.0, ge=1.0, le=100.0)

    # pdf-inspector / hybrid routing.
    native_pdf_enabled: bool = True
    native_failure_policy: NativeFailurePolicy = NativeFailurePolicy.PDFIUM_THEN_OCR
    native_min_text_chars: int = Field(default=5, ge=0)
    native_keep_short_text_pages: bool = True

    # Nạp state_dict VietOCR bằng torch.load(weights_only=True).
    vietocr_safe_weights_only: bool = True

    # API/output.
    include_page_markers: bool = True
    include_markdown_in_response: bool = True
    include_page_results: bool = True
    warm_models_on_startup: bool = False
    log_level: str = "INFO"
    log_format: str = Field(default="compact", pattern="^(compact|json)$")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolve_project_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    @property
    def paddle_device(self) -> str:
        if self.device.startswith("cuda"):
            index = self.device.split(":", 1)[1] if ":" in self.device else "0"
            return f"gpu:{index}"
        return self.device

    @property
    def vietocr_device(self) -> str:
        if self.device.startswith("gpu"):
            index = self.device.split(":", 1)[1] if ":" in self.device else "0"
            return f"cuda:{index}"
        return self.device
