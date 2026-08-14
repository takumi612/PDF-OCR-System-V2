# Pipeline chi tiết — Government OCR Text API 1.3.5.6

## 1.3.5.6: selective evidence + CPU runtime tuning

Bản 1.3.5.6 giữ visual grounding/cross-segment của 1.3.5.5 nhưng không còn trace decoder cho mọi segment. Mỗi trang chọn tối đa 8 risk seed; nếu seed nằm trên một line đã split thì các segment cùng line chỉ được trace làm anchor. Context-only trace không được local-mutate. CPU runtime được cấu hình rõ ràng trước model load, Paddle detector thử oneDNN/MKL-DNN với fallback an toàn, và crop blank/rule được lọc trước sequence recognizer.


## 1. Phạm vi

Đầu vào là PDF; đầu ra là text UTF-8 và Markdown theo trang. Bounding box, crop và
attention chỉ tồn tại nội bộ. Nhánh native text không thay đổi; 1.3.5.6 tập trung
vào giảm recognition overhead mà vẫn giữ visual validation có mục tiêu.

## 2. Luồng tổng thể

```text
PDF
  → pdf-inspector / PDFium routing
      ├─ text layer hợp lệ → native extraction
      └─ scan / image-only
           → render 200 DPI
           → quality gate
           → PP-OCRv6_medium_det
           → line crop + layout ordering
           → width-cap split baseline 1.3.3; overlap tắt
           → conservative blank/rule crop filter
           → adaptive VietOCR greedy batch
           → selective decoder probability + attention trace
           → visual coverage gain + attention reuse
           → map attention về line-coordinate gốc
           → suffix/midline/cross-segment delete-only validation
           → optional multi-view fallback (mặc định tắt)
           → loop/tail validation
           → join segment → page text
```

## 3. Nguyên tắc chống overfit theo hai PDF

Semantic guard không có:

- blacklist từ;
- dictionary sửa lỗi;
- regex chứa artifact của corpus;
- bảng thay thế text;
- LLM viết lại câu.

Phép biến đổi duy nhất là **delete-only**. Nội dung bên trái và bên phải span được
giữ nguyên thứ tự. Test runtime quét source để ngăn các artifact regression bị
đưa vào semantic code. Test lexical-invariance áp cùng evidence lên nhiều nội dung
không liên quan và token sinh ngẫu nhiên.

Xem `CORPUS_COUPLING_AUDIT.md` và chạy:

```powershell
python scripts/audit-corpus-coupling.py --terms-file .\runtime\observed-artifacts.txt
```

Công cụ audit không có danh sách term tích hợp; dữ liệu đánh giá được cấp từ ngoài.

## 4. Decoder evidence

VietOCR `vgg_seq2seq` được trace theo contract thực:

```text
prediction, next_hidden, attention = decoder(input_token, hidden, encoder_outputs)
```

Với mỗi ký tự, pipeline ghi:

- token probability;
- attention center theo trục X;
- attention spread;
- attention-weighted ink support;
- vị trí vùng mực cuối của ảnh.

Trace chỉ được dùng khi text trace sau normalize khớp text greedy. Mismatch hoặc
contract error giữ nguyên output.

## 5. Selective semantic coverage

Mặc định 1.3.5.6:

```text
DECODER_EVIDENCE_FULL_COVERAGE=false
DECODER_EVIDENCE_MAX_CHECKS_PER_PAGE=8
DECODER_EVIDENCE_INCLUDE_SPLIT_LINE_CONTEXT=true
```

Risk score chọn tối đa 8 **seed** mỗi trang. Với line bị split, toàn bộ segment cùng line được trace để cung cấp right/left anchor cho cross-segment guard; những context-only segment này không chạy local mutation. Có thể bật lại full coverage để A/B hoặc điều tra false negative.

## 6. Suffix validation

Suffix có thể bị loại khi:

1. token probability và ink support cùng thấp so với phần ổn định của segment;
2. attention stall, near-loop hoặc đã tới cuối bằng chứng mực;
3. prefix giữ lại đủ dài;
4. span số không được tự động áp dụng.

Attention-cluster expansion có thể lùi điểm cắt về đầu từ trước đó khi toàn bộ từ
cùng cluster attention yếu. Không kiểm tra từ đó có nghĩa hay không.

## 7. Unsupported midline span

Midline span là một cụm yếu nằm giữa hai phần được ảnh hỗ trợ. Một span chỉ được
đề xuất khi đồng thời đạt các điều kiện sau:

- có ít nhất một word anchor bên trái và bên phải;
- cả hai anchor có probability và ink support cao hơn span;
- span yếu tương đối so với cả hai anchor, không chỉ thấp theo ngưỡng tuyệt đối;
- attention trong span có range/progress nhỏ hoặc quay lại cluster lân cận;
- attention sau span phục hồi theo chiều đọc;
- span không chiếm quá tỷ lệ cấu hình của segment;
- numeric span vẫn warning-only.

Pseudo-flow:

```text
supported left anchor
        ↓
weak word span + stalled/recurrent attention
        ↓
supported right anchor + forward recovery
        ↓
delete only the weak span
```

Nếu right anchor không được hỗ trợ hoặc attention không phục hồi, pipeline không
xóa giữa câu.

## 8. Line-level visual grounding và cross-segment validation

Mỗi decoder step lưu thêm ba tín hiệu không phụ thuộc từ vựng:

- `novel_ink_support`: lượng mực mới mà attention vừa chạm tới;
- `visual_coverage_gain`: tỷ lệ visual evidence mới so với phần đã được đọc;
- `reused_attention_ratio`: tỷ lệ attention quay lại vùng encoder đã được dùng.

Vị trí attention local của segment được ánh xạ về line crop gốc bằng
`source_left/source_right`. Khi một line bị chia nhiều segment, head của segment
sau có thể làm right anchor cho tail của segment trước:

```text
supported left text → weak/reused tail | strong next-segment head
                                      ↓
                           cross_segment_visual_gap
                                      ↓
                                 delete-only
```

Cross-segment guard yêu cầu từng word trong span đều yếu theo visual support,
coverage gain và probability tương đối; một từ được ảnh hỗ trợ sẽ chặn việc mở
rộng span về bên trái. Numeric span vẫn chỉ cảnh báo.

Boundary bị từ chối được ghi trong `decoder_evidence_cross_segment_rejections`
để phân tích điều kiện thất bại trên PDF thật.

### Secondary verifier và semantic-safe output

`SECONDARY_RECOGNIZER_ENABLED=true` là mặc định và khởi tạo PaddleOCR
`TextRecognition` với `latin_PP-OCRv5_mobile_rec`. Verifier chạy theo batch trên
các line crop cuối cùng. Nếu oneDNN/PIR gặp lỗi tương thích đã biết, model được
rebuild với `enable_mkldnn=false` và retry đúng một lần.

Secondary output chỉ là bằng chứng so sánh; nó không được phép chèn hoặc thay
token. Auto-change duy nhất là cắt suffix không chứa chữ số khi prefix đủ dài,
primary đang rủi ro và secondary confidence đạt ngưỡng. Mọi bất đồng số, retry
chứa số, dòng trống hoặc bất đồng chưa đủ điều kiện xóa đều giữ nguyên text và
được đánh dấu high-risk.

Response tách hai kênh: `text`/`raw_text` để audit và `ai_safe_text` cho hệ thống
AI phía sau. Dòng high-risk được thay bằng marker `OCR_SEMANTIC_RISK`; downstream
không được dùng `text` làm đầu vào mặc định. `SECONDARY_RECOGNIZER_APPLY_CHANGES`
vẫn là biến legacy và không cấp quyền rewrite.

## 9. Telemetry

Các metrics chính:

```text
decoder_evidence_candidate_count
decoder_evidence_seed_selected_count
decoder_evidence_context_forced_count
decoder_evidence_selected_count
decoder_evidence_unchecked_candidate_count
decoder_evidence_supported_count
decoder_evidence_trace_mismatch_count
decoder_evidence_midline_span_count
decoder_evidence_midline_trimmed_count
decoder_evidence_line_evidence_count
decoder_evidence_cross_segment_candidate_count
decoder_evidence_cross_segment_trimmed_count
decoder_evidence_cross_segment_rejected_count
decoder_evidence_visual_coverage_exhausted_count
decoder_evidence_attention_reuse_count
secondary_verifier_count
secondary_verifier_primary_extra_count
secondary_verifier_conflict_count
decoder_evidence_trimmed_count
decoder_evidence_suspicious_numeric_count
decoder_evidence_ms
```

Mỗi event có thể chứa:

```json
{
  "raw_text": "...",
  "proposed_text": "...",
  "validated_text": "...",
  "applied": true,
  "removed_text": "...",
  "reason": "unsupported_midline_span",
  "span_kind": "midline",
  "span_start": 24,
  "span_end": 51,
  "span_probability_mean": 0.28,
  "span_ink_support_mean": 0.03,
  "span_attention_range": 0.05,
  "span_attention_progress": 0.01,
  "left_anchor_probability_mean": 0.91,
  "left_anchor_ink_support_mean": 0.65,
  "left_anchor_attention_mean": 0.42,
  "right_anchor_probability_mean": 0.88,
  "right_anchor_ink_support_mean": 0.61,
  "right_anchor_attention_mean": 0.72
}
```

## 10. Các rule cũ đã audit

- Split/merge: hình học và projection mực, không lexical.
- Decoder-loop: n-gram/character repetition, không chứa phrase corpus.
- Multi-view consensus: delete-only subsequence, mặc định tắt.
- Signature ordering: bbox geometry; các cụm chức danh trong comment không được
  đọc khi chạy.
- Structural heading retry: domain-specific nhưng label đã cấu hình được qua
  `CHAPTER_HEADING_RETRY_LABELS`; label được regex-escape.

## 11. Cấu hình mặc định 1.3.5.6

```text
DECODER_EVIDENCE_ENABLED=true
DECODER_EVIDENCE_APPLY_CHANGES=true
DECODER_EVIDENCE_FULL_COVERAGE=false
DECODER_EVIDENCE_MAX_CHECKS_PER_PAGE=8
DECODER_EVIDENCE_INCLUDE_SPLIT_LINE_CONTEXT=true
DECODER_EVIDENCE_MIDLINE_ENABLED=true
DECODER_EVIDENCE_NUMERIC_APPLY_CHANGES=false
OCR_NONTEXT_CROP_FILTER_ENABLED=true
CPU_RUNTIME_TUNING_ENABLED=true
TORCH_CPU_THREADS=4
TORCH_INTEROP_THREADS=1
PADDLE_CPU_THREADS=4
PADDLE_ENABLE_MKLDNN=true
HALLUCINATION_GUARD_ENABLED=false
EXPERIMENTAL_OVERLAP_SPLIT=false
TAIL_SEGMENT_RETRY_ENABLED=false
```

## 12. Xác minh đóng gói

```text
python -m compileall -q src scripts tests
python -m pytest -q
```

Kết quả môi trường đóng gói 1.3.5.6: `101 passed, 2 skipped`.

Môi trường build không có checkpoint Paddle/VietOCR thật, nên chưa khẳng định
CER/WER hoặc hiệu quả trên hai PDF. Lần chạy thực phải kiểm tra toàn bộ
`decoder_evidence_events`, đặc biệt các event `span_kind=midline`.
