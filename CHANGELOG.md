# Changelog

## 1.3.5.9 - 2026-08-09

- Thêm semantic retry crop theo trục ngang ở nhiều độ rộng và chỉ nhận kết quả khi VietOCR/Tesseract/PP-OCRv5 đạt đồng thuận bảo thủ.
- Bảo vệ tuyệt đối token chứa số/mã pháp lý; mọi bất đồng numeric từ verifier hoặc recognizer phụ đều từ chối rewrite.
- Thêm khôi phục dấu câu/khoảng trắng an toàn, làm sạch dấu lặp và tiền tố OCR bẩn, chặn dấu nháy giả cuối dòng.
- Thêm đồng thuận punctuation ba engine cho dòng không cần resize-retry; token surface của OCR chính được giữ nguyên.
- Thêm chuẩn hóa chính tả pháp lý ngữ cảnh hẹp và audit `raw_text`/`semantic_reasons` cho từng thay đổi.
- Tăng quota retry lên toàn bộ dòng đủ điều kiện trong corpus kiểm thử và bổ sung telemetry eligible/attempted/applied/surface-consensus.
- Benchmark offline CPU 5 PDF scan, 19 trang, 710 dòng: 5/5 lỗi nặng đã biết đúng; vòng cuối 731.922 giây; 0 hit trên các mẫu lỗi chính tả/hallucination đã phát hiện.
- Bổ sung test hồi quy cho số pháp lý, dấu câu, dấu nháy, list item, collocation và đồng thuận ba engine; 149 test semantic/integration liên quan vượt qua.

## 1.3.5.8 - 2026-08-09

- Tắt oneDNN mặc định trên Windows/Paddle 3.3.1 để tránh lỗi PIR `ConvertPirAttribute2RuntimeAttribute` và giữ fallback khi chủ động bật.
- Thêm Tesseract `vie+eng/tessdata_best` làm verifier toàn trang và định tuyến PP-OCRv5 có chọn lọc.
- Thêm `ai_safe_text`, `ai_ready`, `semantic_risk` và telemetry bất đồng semantic.

## 1.3.5.6 - 2026-08-07

- Chuyển decoder evidence mặc định từ full coverage sang **selective top-K**: `DECODER_EVIDENCE_FULL_COVERAGE=false`, tối đa 8 seed candidate/trang. Mục tiêu là bỏ chi phí trace VietOCR lần hai trên mọi segment.
- Thêm **split-line context closure**: khi một seed thuộc line bị split, toàn bộ segment của line được trace làm visual anchor cho cross-segment validation nhưng context-only segment không được tự ý local-trim.
- Thêm telemetry `decoder_evidence_seed_selected_count` và `decoder_evidence_context_forced_count` để tách risk seed khỏi context trace.
- Thêm **CPU runtime tuning** trước khi model load: mặc định Torch intra-op 4 thread, inter-op 1; không sửa OMP/MKL env toàn cục. Runtime log ghi thread trước/sau.
- Paddle detector mặc định thử `enable_mkldnn=true`, `cpu_threads=4`; nếu oneDNN/MKL-DNN khởi tạo lỗi thì tự fallback về `enable_mkldnn=false`. `detector_runtime` ghi requested/effective/fallback.
- Thêm **conservative non-text crop filter** trước VietOCR cho crop gần như blank hoặc horizontal rule/dotted leader; chỉ dùng pixel geometry, không dùng từ/cụm từ. Metrics ghi input/kept/filtered và reason counts.
- Sửa dedup correctness: không còn drop hai dòng liên tiếp chỉ vì text giống nhau. Chỉ suppress duplicate khi text bằng nhau **và bbox overlap IoU >= 0.80**.
- Thêm greedy batch fallback telemetry (`greedy_batch_fallback_*`) để phát hiện `predict_batch()` âm thầm rơi về sequential inference.
- Cập nhật analyzer/compare script cho selective evidence, fallback và crop-filter metrics.
- Giữ secondary PP-OCRv5 verifier tắt mặc định, numeric delete warning-only, splitter/layout/checkpoint/API schema không đổi.
- Kết quả build source package: `101 passed, 2 skipped` trong môi trường không cài runtime VietOCR/Paddle integration đầy đủ.

## 1.3.5.5 - 2026-08-07

- Sửa visual-grounding metric bị collapse trên attention thật: bỏ absolute bin threshold `0.08`, dùng continuous attention overlap và incremental visual mass nên hoạt động với attention diffuse và không phụ thuộc encoder length.
- Thêm A/B switch `DECODER_EVIDENCE_VISUAL_GROUNDING_ENABLED`; kết hợp với `DECODER_EVIDENCE_CROSS_SEGMENT_ENABLED` để tách chi phí baseline / grounding / cross-segment trong cùng một build.
- Thêm performance diagnostics bắt buộc trong giai đoạn semantic: wall time vs process CPU time, CPU frequency, RSS/peak RSS, process/thread-pool metadata và unaccounted wall time.
- Tách decoder-evidence timing thành preprocess, encoder, decoder forward, attention extraction, torch postprocess, visual-grounding NumPy và trace construction; cross-segment analysis có timer riêng.
- Thêm workload fingerprint gồm segment/batch/pixel count, decoder forward/sample-step count, attention element count và trace character count để phát hiện runtime regression khi workload không đổi.
- Rejected cross-segment events ghi raw tail/head probability/support/coverage/reuse thay vì chỉ `failed_conditions`.
- Thêm `scripts/compare-ocr-performance.py` để so hai structured log và chỉ ra stage, workload fingerprint, CPU/frequency/RSS khác nhau.
- Thêm realistic diffuse-attention tests ở source length 64/128/256; tổng build: 95 passed, 2 skipped trong môi trường không cài runtime VietOCR integration đầy đủ.
- Không thay detector, splitter, layout, checkpoint, numeric warning-only policy hoặc secondary-recognizer default.

## 1.3.5.4 - 2026-08-07

- Thêm visual-grounding signal theo decoder step: `novel_ink_support`, `visual_coverage_gain` và `reused_attention_ratio`; không phụ thuộc token identity.
- Ánh xạ attention của split segment về tọa độ X của line crop gốc để phân tích continuity xuyên ranh giới segment.
- Thêm `cross_segment_visual_gap`: dùng head có visual support của segment kế tiếp làm right anchor để xác nhận và xóa weak tail ở segment trước; vẫn delete-only và numeric warning-only.
- Cross-segment span được đánh giá theo từng word để không kéo điểm cắt ngược vào một word có pixel support tốt.
- Thêm telemetry rejected boundary và metrics line/cross-segment/coverage/reuse để phân tích false negative trên PDF thật.
- Thêm optional PP-OCRv5 Latin `TextRecognition` verifier. Mặc định tắt; secondary OCR không rewrite primary và chỉ có thể gate một delete-only proposal khi được bật rõ ràng.
- Bổ sung test visual coverage, attention reuse, global line mapping, cross-segment lexical invariance, supported-tail protection, numeric safety và secondary conflict. Kết quả build: 91 passed, 2 skipped.
- Không thay đổi detector, splitter baseline 1.3.3, layout, native PDF fast path, checkpoint VietOCR hoặc API response schema.

## 1.3.5.3 - 2026-08-07

- Audit toàn bộ runtime từ 1.3.0 đến 1.3.5.2: không có blacklist, correction dictionary hoặc bảng thay thế chứa artifact của hai PDF. Một cụm corpus chỉ xuất hiện trong docstring 1.3.5.2 và đã được loại khỏi runtime.
- Thêm `CORPUS_COUPLING_AUDIT.md` và `scripts/audit-corpus-coupling.py`; công cụ audit nhận danh sách term từ bên ngoài, không tự chứa blacklist.
- Bật semantic full coverage: mọi segment có text được trace khi `DECODER_EVIDENCE_FULL_COVERAGE=true`; giá trị `MAX_CHECKS_PER_PAGE=0` có nghĩa không giới hạn.
- Thêm phát hiện `unsupported_midline_span`: chỉ xóa span giữa hai anchor được ảnh hỗ trợ khi probability/ink support suy giảm, attention stall hoặc recurrence, và attention phục hồi về phía trước sau span.
- Midline rule không đọc từ/cụm từ. Thêm lexical-invariance tests và 25 bộ token ngẫu nhiên để chứng minh cùng evidence cho cùng quyết định bất kể nội dung từ vựng.
- Numeric span tiếp tục warning-only. Không thêm, thay thế hoặc sửa chính tả token; phép biến đổi duy nhất vẫn là delete-only có evidence.
- Structural-heading retry không còn khóa cứng vào một nhãn trong logic; nhãn được cấu hình qua `CHAPTER_HEADING_RETRY_LABELS` và được regex-escape.
- Telemetry mới: `decoder_evidence_midline_span_count`, `decoder_evidence_midline_trimmed_count`, span offsets, anchor probability/ink/attention và span progression.
- Kết quả môi trường đóng gói: 81 passed, 2 skipped. Chưa chạy checkpoint OCR thật trong môi trường build.

## 1.3.5.2 - 2026-08-07

- Trace decoder evidence theo batch cho các ảnh đã được padding cùng kích thước; tăng coverage mặc định từ 6 lên 12 candidate/trang mà không chạy lại encoder cho từng dòng.
- Thêm ngưỡng probability/ink support tương đối theo chất lượng prefix của từng segment; vẫn giữ giới hạn tuyệt đối và cap để tránh nới ngưỡng quá mức.
- Mở rộng điểm trim về ranh giới từ khi từ ngay trước suffix cùng attention cluster, có ink support yếu và nằm gần cuối bằng chứng ảnh. Không dùng blacklist/từ điển.
- Giữ dấu câu tại seam khi xóa suffix, tránh làm mất dấu phẩy/chấm phẩy hợp lệ.
- Multi-view retry mặc định tắt vì benchmark thực tế có chi phí nhưng chưa tạo consensus; vẫn có thể bật bằng cấu hình.
- Numeric evidence vẫn warning-only. Event log tách rõ `proposed_text`, `validated_text` và `applied`.
- Thêm metrics coverage, batch trace và cluster expansion; cập nhật analyzer.
- Bổ sung regression tests cho partial-word trim, relative thresholds, batch decoder contract, coverage metrics và numeric warning event.


## 1.3.5.1 - 2026-08-07

- Sửa adapter attention trace theo đúng contract của VietOCR `vgg_seq2seq`: decoder nhận một token + hidden state và trả `prediction, next_hidden, attention`.
- Thêm telemetry lỗi trace: `decoder_evidence_trace_error_count`, `decoder_evidence_trace_error_types`, `decoder_evidence_disabled_reason` và circuit breaker theo tài liệu.
- Không còn nuốt exception im lặng; lỗi kiến trúc/contract được ghi audit và chỉ thử một lần thay vì lặp trên mọi segment.
- Thay test mock sai contract bằng decoder giả bám đúng shape/state progression của VietOCR.
- Thêm test EOS/special token, attention 2D/3D, malformed decoder, mismatch không mutate, false-positive guard, legal repeated phrase và circuit breaker.
- Thêm integration test không cần checkpoint cho class `Seq2Seq` thật của VietOCR 0.3.13; test tự skip nếu package chưa được cài trong môi trường build.
- Kết quả môi trường đóng gói: 61 passed, 1 skipped; không thay dependency, checkpoint, splitter, layout hoặc API schema.

## 1.3.5 - 2026-08-06

- Rollback width-cap split và line-detector ordering về đúng baseline 1.3.3; loại bỏ morphology seam và provisional signature cluster đã làm tăng segment/regression ở 1.3.4.
- Thêm decoder-evidence guard cho VietOCR `vgg_seq2seq`: đọc probability và attention trực tiếp từ decoder đang nạp, không sửa checkpoint hay package VietOCR.
- Ánh xạ attention theo từng ký tự vào projection mực của đúng ảnh đã đưa vào batch, bao gồm common-width padding.
- Chỉ tự động cắt suffix khi attention stall, visual evidence exhausted hoặc near-loop có bằng chứng attention; không sửa insertion giữa câu.
- Numeric span chỉ được cảnh báo, không tự động xóa mặc định.
- Thu hẹp multi-view guard thành suffix-only, tối đa 4 recheck/trang; tắt prefix/midline/numeric mutation.
- Thêm metrics/audit `decoder_evidence_*` và error code `decoder_attention_suffix_trimmed`.
- Thêm regression test cho attention trace, suffix evidence, numeric warning-only và near-loop biến thể; tổng cộng 52 test pass.
- Không thay đổi dependency, checkpoint, native text fast path hoặc API schema.

## 1.3.4 - 2026-08-06

- Thêm multi-view hallucination guard không dùng từ điển/LLM: nhận dạng lại segment rủi ro trên crop sát mực hoặc dominant-ink.
- Chỉ cho phép phép biến đổi delete-only khi output retry là subsequence của output gốc; không thay thế hoặc tự bổ sung token.
- Hỗ trợ loại insertion ở prefix, suffix và giữa dòng khi có anchor hai phía, confidence phù hợp và bằng chứng blank/crop geometry.
- Bảo vệ token số bằng ngưỡng hình học nghiêm ngặt hơn; các hậu tố số chỉ bị loại khi crop retry thay đổi đủ lớn và confidence không giảm.
- Thêm morphology-aware split seam: bảo vệ vùng quanh cột có mực, ưu tiên khoảng trắng thật thay vì valley hẹp giữa nét ký tự.
- Sửa `signature_block` theo provisional right cluster trước khi tính boundary; cặp chức danh crossing không còn bị đẩy vào prefix.
- Thêm audit events và metrics `hallucination_guard_*`, `signature_cluster_count`.
- Thêm 6 regression test mới; tổng cộng 49 test pass.
- Không thay đổi dependency, checkpoint, native text fast path hoặc API schema.

## 1.3.3 - 2026-08-06

- Sửa phân loại crossing polygon trong `signature_block`: dòng compact như `KT. THỦ TƯỚNG` được gán theo cụm anchor phải thay vì chỉ dùng tâm bbox.
- Giữ các dòng `Nơi nhận` dài bắt đầu từ lề trái ở cột trái dù bbox cắt gutter.
- Xếp polygon con dấu/chữ ký tay lớn sau các dòng chữ ký compact; không để bbox trang trí chen trước `KT./PHÓ THỦ TƯỚNG`.
- Thêm targeted chapter-heading retry: mở crop sang phải khi output chỉ là `Chương`; chỉ chấp nhận đúng mẫu tiêu đề chương có số La Mã/Ả Rập.
- Retry chương thất bại giữ nguyên text và gắn `chapter_heading_incomplete`, không tự sửa bằng từ điển.
- Bổ sung metrics `signature_crossing_to_right_count`, `signature_decorative_polygon_count` và nhóm `chapter_heading_retry_*`.
- Thêm 4 regression test mới; tổng cộng 43 test pass.
- Không thay đổi dependency, checkpoint, width-cap split, tail retry policy hoặc native fast path.

## 1.3.2 - 2026-08-06

- Sửa boundary của `signature_block`: dùng top-y của cột phải làm điểm bắt đầu layout hai cột thay vì min-y của cả hai cột.
- Giữ nguyên row-order cho phần thân bài kết thúc trước `signature_boundary_y`; chỉ reorder suffix thành cột trái rồi cột phải.
- Giữ đúng thứ tự `Điều 4` khoản 1, khoản 2 trước khối `Nơi nhận` trên hình học sát trang 5 `01-bct.signed.pdf`.
- Giữ nguyên cải thiện trang 9 `30-ttg.signed.pdf`: toàn bộ `Nơi nhận` trước khối chữ ký.
- Bổ sung metrics `signature_boundary_y`, `signature_prefix_count`.
- Thêm regression test cho body prefix sát khối chữ ký và schema metrics ở layout cột thường.
- Không thay đổi VietOCR, width-cap split, tail retry policy, dependency hoặc native fast path.

## 1.3.1 - 2026-08-06

- Tắt overlap ảnh theo mặc định sau khi benchmark thật cho thấy CER/WER tăng mạnh do seam merge không khử được các dự đoán khác nhau trong vùng chồng lấn.
- Thêm `EXPERIMENTAL_OVERLAP_SPLIT=false`; giá trị overlap cũ trong `.env` không còn tác động nếu cờ thử nghiệm chưa bật.
- Tắt tail retry thay đổi text theo mặc định; vẫn giữ metrics và cảnh báo `tail_segment_uncertain`.
- Giới hạn khối chữ ký bên phải tối đa 6 dòng.
- Xem hàng có từ 2 ô trở lên là tín hiệu bảng/biểu mẫu để tránh reorder sai form hai cột.
- Bổ sung metrics `column_left_counts` và `column_right_counts`.


## 1.3.0 - 2026-08-06

- Split crop width-cap có overlap pixel quanh seam; giữ từ ngắn sát điểm cắt.
- Ghép segment bằng suffix-prefix token/character overlap thay vì nối mù.
- Tail-segment validation: trim blank, retry greedy có ngân sách, đánh dấu hoặc loại hallucination cực ít mực.
- Phát hiện decoder loop dạng hai n-gram đầy đủ cộng prefix lần thứ ba ở cuối output.
- Thay union-gutter bằng occupancy-gutter; cho phép một số polygon dài/con dấu đi qua vùng giữa cột.
- Thêm mode `signature_block` cho khối `Nơi nhận` nhiều dòng bên trái và chữ ký thưa bên phải.
- Giữ tạm polygon cao-hẹp 3--7 px để gộp `Chương I/III` hoặc marker cùng baseline; fragment cô lập bị loại sau merge.
- Thêm metrics seam, tail, partial-loop, occupancy, crossing, signature mode và fragment merge/drop.
- Thêm regression test dùng hình học đo từ ảnh render trang 9 QĐ 30/2025, test overlap seam, tail validation, partial loop và ký tự La Mã hẹp.
- Không thay đổi dependency, checkpoint hoặc native text fast path.

## 1.3.0 - 2026-08-06

- Thêm overlap pixel quanh mọi seam khi chia crop width-cap; core segment vẫn nằm trong giới hạn `image_max_width=512`.
- Ghép segment bằng exact suffix-prefix token/character overlap, tránh lặp hoặc mất từ ngắn như `an` tại seam.
- Thêm tail-segment validation độc lập với `PAD_BATCHES_TO_COMMON_WIDTH`: trim blank phải và retry greedy có giới hạn; segment không có mực nhưng sinh text bị loại và gắn review code.
- Phát hiện decoder loop dạng hai n-gram đầy đủ cộng prefix của lần thứ ba ở cuối output.
- Thay gutter union tuyệt đối bằng occupancy gutter, bỏ polygon con dấu/bbox bất thường khỏi bước tìm gutter và hỗ trợ khối chữ ký bất cân bằng.
- Thêm same-baseline fragment merge cho `Chương I/III`, marker `a)`/`b)` khi detector tách ký tự hẹp thành polygon riêng.
- Thêm table-like row guard để biểu mẫu nhiều ô cùng hàng không bị đọc như hai cột văn bản.
- Bổ sung metrics seam, tail retry/suppression, partial loop, occupancy/crossing và baseline fragment merge.
- Thêm test mô phỏng hình học thực tế trang 9 QĐ 30/2025 có dòng dài lấn gutter và bbox con dấu lớn.
- Tổng cộng 33 test pass; không thay đổi checkpoint, dependency hoặc nhánh text-layer.

## 1.2.0 - 2026-08-06

- Xử lý gốc width-cap: thử chia mọi crop vượt `image_max_width=512` tại run khoảng trắng giữa từ.
- Không cưỡng bức cắt; crop không có điểm cắt an toàn được giữ nguyên và gắn `width_cap_unresolved`.
- Thêm column-aware reading order cho OCR scan bằng vertical block + gutter detection.
- Giữ text của dòng có quality error nhưng đánh dấu `needs_review`, không làm mất cả dòng.
- Gắn `decoder_loop_trimmed` khi heuristic đã cắt hallucination để bắt buộc đối chiếu bản gốc.
- Thêm metrics width-cap và layout ordering.
- Thêm regression tests cho long-line split, unresolved width-cap, two-column signature block và short-signature scan routing.

## 1.1.2 - 2026-08-06

- Điều chỉnh ngưỡng split an toàn từ 2.5 xuống 2.0 lần model width sau benchmark 1.1.1.
- Giữ nguyên nguyên tắc chỉ cắt tại valley ít mực; không cưỡng bức cắt giữa chữ.
- Phát hiện và cắt chuỗi ký tự lặp dài như `000000...` và chuỗi dấu câu lặp.
- Khi token/n-gram loop bắt đầu ngay đầu dòng hoặc chỉ có một token nhiễu phía trước, bỏ phần dòng lỗi thay vì giữ nguyên hallucination.
- Không coi hai lần lặp n-gram là lỗi để tránh cắt nhầm cụm hợp lệ như `đại diện/đại diện`.
- Bổ sung metrics `decoder_char_loop_detected_count` và `decoder_char_loop_trimmed_count`.
- Thêm regression tests cho loop đầu dòng, chuỗi số lặp, dấu câu lặp và cụm lặp hợp lệ.

## 1.1.1 - 2026-08-06

- Điều chỉnh split crop sau benchmark thực tế: ngưỡng từ 1.5 lên 2.5 lần model width.
- Chỉ cắt tại valley ít mực; bỏ cưỡng bức cắt giữa dòng.
- Giới hạn tối đa 4 segment mỗi crop.
- Tăng recognition pixel budget lên 12288 và width ratio lên 2.0.
- Giảm beam retry mặc định còn 1 lần mỗi trang.
- Bỏ import trực tiếp `pkg_resources` trong code kiểm tra để không tự phát cảnh báo deprecation.
- Thêm data-URI favicon để trình duyệt không gọi `/favicon.ico`.
- Thêm regression test bảo đảm crop rộng vừa phải không bị chia.

## 1.1.0

- Gom OCR theo recognition window nhiều trang, mặc định 3 trang.
- Pad ảnh trong cùng batch về exact common width để tránh VietOCR tự chia lại theo tensor width.
- Thay 5 width bucket cố định bằng adaptive batch scheduler theo resized width, pixel budget và width ratio.
- Chia crop quá dài tại valley ít mực trước VietOCR để loại width-cap.
- Chỉ beam retry decoder loop thật sự và giới hạn tối đa 2 retry mỗi trang.
- Phát hiện repeated n-gram 1–4 token; trim loop vượt ngân sách retry khi còn prefix hợp lệ.
- Dùng `torch.inference_mode()` cho mọi đường inference.
- Bổ sung page/window metrics cho split, batch, greedy, beam và decoder loop.
- Thêm regression tests cho wide-crop split, adaptive batching, retry budget và document window.
- Không thay đổi native text fast path hoặc dependency.

## 1.0.4

- Sửa routing PDF có text layer: không ép trang trắng/trang chỉ có vài ký tự sang OCR.
- Khi `pdf-inspector` lỗi, dùng text layer PDFium theo từng trang trước khi quyết định OCR, thay vì OCR toàn bộ tài liệu.
- Thêm log `native_routing_complete` với danh sách trang OCR và lý do routing.
- Thêm metrics `pages_needing_ocr`, `ocr_reasons`, `native_fallback_reason`.
- Nạp checkpoint VietOCR bằng `torch.load(weights_only=True)` để loại FutureWarning và hạn chế unpickle đối tượng tùy ý.
- Thêm regression tests cho short native page, blank page, PDFium fallback và safe checkpoint loading.

## 1.0.3

- Khóa chính xác `Pillow==10.2.0` theo dependency contract của `vietocr==0.3.13`.
- Sửa installer Windows và Dockerfile để phục hồi Pillow 10.2.0 sau khi cài Torch/Torchvision.
- Thêm `pip check` vào luồng cài đặt và hướng dẫn xử lý xung đột Pillow trong README.
- Bổ sung kiểm tra phiên bản Pillow sau cài đặt.

## 1.0.2

- Bổ sung dependency bắt buộc `torchvision`; VietOCR dùng `torchvision.models` cho backbone VGG.
- Khóa cặp CPU tương thích `torch==2.5.1` và `torchvision==0.20.1`.
- Hạ pin `setuptools` xuống `80.10.2` để tiếp tục cung cấp `pkg_resources` cho `gdown==4.4.0`.
- Cập nhật README, Dockerfile và `scripts/install-cpu.ps1` với lệnh cài PyTorch từ CPU wheel index chính thức.
- Thêm bước kiểm tra import Torch/Torchvision/pkg_resources sau cài đặt.

## 1.0.1

- Khóa `setuptools==81.0.0` để duy trì `pkg_resources` cho `vietocr==0.3.13`/`gdown==4.4.0`.
- Sửa README, PowerShell installer, Dockerfile và build-system để không nâng Setuptools lên 82+.
- Thêm lỗi hướng dẫn rõ ràng khi môi trường thiếu `pkg_resources`.

## 1.0.0

- Tách standalone API đồng bộ `POST /api/extract`.
- Kế thừa PP-OCRv6 line detector + VietOCR vgg_seq2seq.
- Kế thừa batch width buckets, beam retry, quality gate và tensor cleanup policy.
- Thêm pdf-inspector để route/trích xuất native page.
- Giữ PDF session mở một lần, không lưu ảnh/bounding box/artifact.
- Thêm giao diện so sánh PDF gốc và text/Markdown.
- Thêm README, PIPELINE, requirements, Dockerfile, PowerShell scripts và unit tests.
## 0.1.2

- Sửa README Windows: giải nén ZIP không tạo thư mục lồng, xác minh project root.
- Bổ sung cách chạy trực tiếp bằng `python -m uvicorn` và `scripts/run.cmd`.
- Bổ sung hướng dẫn xử lý PowerShell Execution Policy và cache endpoint `/api/jobs`.
- Script sao chép VietOCR tự tạo thư mục đích nếu chưa tồn tại.
