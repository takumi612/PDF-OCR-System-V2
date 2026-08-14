# Government OCR Text API 1.3.5.9

API standalone nhận một file PDF và trả văn bản UTF-8/Markdown. Nhánh PDF có
text layer vẫn dùng `pdf-inspector`; 1.3.5.9 bổ sung kiểm chứng semantic độc lập,
định tuyến PP-OCRv5 có chọn lọc và một kênh text an toàn cho hệ thống AI phía sau.

Backend không xuất ảnh trang, bounding box, polygon, SVG hay viewer artifact. Giao
diện web hiển thị PDF gốc bên trái và text bên phải để đối chiếu.

## Thay đổi chính trong 1.3.5.9

1. **Retry OCR theo đồng thuận ba nguồn.** Dòng rủi ro được crop lại theo trục ngang ở nhiều độ rộng; chỉ nhận nội dung mới khi VietOCR retry, Tesseract và PP-OCRv5 cùng hỗ trợ, đồng thời giữ nguyên số/mã pháp lý.
2. **Khôi phục cấu trúc bề mặt an toàn.** Dấu câu và khoảng trắng chỉ được sửa khi token không đổi; dấu câu đang có không bị thay bằng loại khác nếu thiếu đồng thuận. Chuỗi bẩn như `; ;`, `. “` và dấu nháy giả cuối dòng bị làm sạch/chặn.
3. **Đồng thuận dấu câu ba engine.** Với dòng không cần resize-retry, punctuation chỉ được thay khi Tesseract và PP-OCRv5 độc lập cùng cho một kết quả và số lượng token khớp.
4. **Chuẩn hóa chính tả pháp lý có ngữ cảnh hẹp.** Sửa các lỗi chắc chắn như `bồ/bố/bỗ sung`, `rủ ro`, `cơ cở`, `giai đoan + năm`, `tai thành viên`, `các tinh, thành phố trực thuộc`, `hợp đồng năm giữ`, `Uu tiên`, `phân bự nguồn lực`; không dùng từ điển thay thế rộng.
5. **Fail-closed vẫn giữ nguyên.** Dòng còn bất đồng cao vẫn mang `semantic_risk=high`, bị che trong `ai_safe_text`, và yêu cầu đối chiếu PDF gốc.

Benchmark CPU/offline cuối trên 5 PDF scan (19 trang, 710 dòng) đạt 5/5 dòng lỗi nặng đã biết đúng chính xác và 0 hit trên bộ mẫu lỗi chính tả/hallucination đã phát hiện. Đây không phải tuyên bố CER/WER tuyệt đối vì corpus chưa có ground truth được chép tay toàn bộ.

## Thay đổi chính trong 1.3.5.8

> Bổ sung hardening CPU/offline ngày 2026-08-09: Tesseract 5.4
> `vie+eng/tessdata_best` chạy một lần trên toàn trang để kiểm chứng độc lập.
> Kết quả Tesseract được ghi vào `verifier_text`/`verifier_confidence` và
> dùng làm bằng chứng; chỉ được sửa text khi có các điều kiện đồng thuận bảo thủ. oneDNN mặc định tắt trên
> Windows/Paddle 3.3.1 để tránh lỗi PIR đã biết; fallback vẫn còn khi chủ động bật.
> Sau lượt Tesseract, PP-OCRv5 chỉ chạy trên dòng có lỗi/độ tin cậy thấp/bất đồng;
> dòng đã đồng thuận được bỏ qua. Nếu tắt Tesseract, hệ thống tự quay về kiểm tra
> mọi dòng để không làm giảm độ phủ an toàn.
> Riêng dòng chứa chữ số/mã văn bản và dòng có bất đồng dấu câu luôn được kiểm
> tra bằng PP-OCRv5, kể cả khi Tesseract không gắn cờ rủi ro.

1. **Không bỏ nhầm dòng văn bản.** Bộ lọc blank/rule chỉ loại crop là một dải mực thật sự mỏng; các dòng luật có nhiều glyph/component được giữ lại.
2. **Không pad chiều rộng mặc định.** `PAD_BATCHES_TO_COMMON_WIDTH=false` tránh làm thay đổi ảnh đầu vào VietOCR và giảm nguy cơ sinh thêm suffix không có trên ảnh.
3. **PP-OCRv5 verifier chạy batch có chọn lọc.** Các dòng rủi ro sau Tesseract được gom xuyên toàn bộ cửa sổ trang rồi mới chạy một lượt batch; recognizer phụ không được quyền chép text của nó vào kết quả.
4. **Sửa tự động có kiểm chứng.** Ngoài guard delete-only cũ, 1.3.5.9 cho phép retry crop và ghép token khi hai OCR độc lập xác nhận; mọi thay đổi số/mã bị từ chối.
5. **Hợp đồng an toàn cho AI.** `text` giữ kết quả đầy đủ để audit; `ai_safe_text` thay dòng rủi ro cao bằng `OCR_SEMANTIC_RISK`; `line_results` giữ raw text, confidence và lý do.

Hệ thống downstream phải dùng `ai_safe_text`. Nếu `ai_ready=false`, cần review các marker trước khi đưa nội dung đó vào mô hình khuyến nghị.

## Thay đổi chính trong 1.3.5.6

1. **Selective decoder evidence mặc định.** Chỉ trace tối đa 8 risk seed/trang thay vì full coverage; có thể bật lại full coverage để A/B.
2. **Split-line context closure.** Seed trên dòng bị split kéo theo các segment cùng line làm visual anchor, nhưng context-only trace không được local-trim.
3. **CPU tuning rõ ràng.** Torch mặc định 4 intra-op / 1 inter-op thread; Paddle detector dùng 4 CPU thread. Không sửa OMP/MKL environment toàn cục.
4. **Paddle oneDNN/MKL-DNN có fallback.** Detector thử `enable_mkldnn=true`; nếu backend lỗi khi khởi tạo hoặc gặp lỗi tương thích oneDNN/PIR đã biết ở lần inference đầu, detector tạo lại backend với `false`, retry đúng một lần và tái sử dụng backend an toàn cho các request sau.
5. **Blank/rule crop filter.** Crop gần như blank hoặc chỉ là horizontal rule/dotted leader được chặn trước VietOCR bằng pixel geometry, giảm hallucination trên form và giảm workload.
6. **Dedup theo geometry.** Hai dòng text giống hệt nhưng ở vị trí khác nhau được giữ; chỉ suppress duplicate khi bbox overlap mạnh.
7. **Batch fallback không còn im lặng.** Log ghi số batch/segment phải rơi từ `predict_batch()` về sequential và loại exception.
8. **Safety giữ nguyên.** Secondary verifier vẫn tắt mặc định; không dictionary/blacklist/LLM rewrite; numeric delete vẫn warning-only.

### Nâng cấp nhanh từ 1.3.5.5

Dừng Uvicorn rồi giải nén patch đè thẳng lên project root. Patch không chứa `.pth`, không đụng `.venv` hoặc model cache.

```powershell
Expand-Archive `
  "D:\Downloads\government-ocr-selective-runtime-patch-1.3.5.6.zip" `
  -DestinationPath "D:\KeySoft\government-ocr-text-api-full" `
  -Force

Set-Location "D:\KeySoft\government-ocr-text-api-full"
$env:PYTHONPATH = "$PWD\src"
python -m compileall -q src scripts tests
python -m pytest -q
```

## Thay đổi chính trong 1.3.5.5

1. **Scale-invariant visual grounding.** Bỏ threshold attention-bin `0.08`; `reused_attention_ratio` dùng continuous overlap `sum(min(current, previous))`, còn visual novelty dùng incremental attention được weighting bởi ink profile.
2. **Performance diagnostics là P0.** Mỗi recognition window ghi wall/process CPU, CPU frequency, RSS, Torch/OMP/MKL thread metadata, accounted/unaccounted wall time và workload fingerprint.
3. **Timing chi tiết decoder.** Tách preprocess, encoder, decoder forward, attention extraction, Torch postprocess, visual-grounding NumPy và trace-build; `cross_segment_analysis_ms` đo riêng.
4. **A/B switch trong cùng build.** `DECODER_EVIDENCE_VISUAL_GROUNDING_ENABLED=false` + `CROSS_SEGMENT_ENABLED=false` cho baseline; bật lần lượt để định vị chính xác regression.
5. **Rejected-boundary evidence.** Rejection event có raw tail/head probability, ink support, coverage gain và attention reuse để calibrate từ trace thật thay vì đoán threshold.
6. **Không đổi safety policy.** Numeric vẫn warning-only; PP-OCRv5 verifier vẫn tắt; không dictionary/blacklist/LLM rewrite.

### Chẩn đoán hiệu năng

Sau khi có hai log structured, chạy:

```powershell
python scripts/compare-ocr-performance.py baseline.log candidate.log
```

Để A/B cùng bản 1.3.5.5, chạy ba cấu hình sau (restart Uvicorn giữa các lần):

```text
A: VISUAL_GROUNDING=false, CROSS_SEGMENT=false
B: VISUAL_GROUNDING=true,  CROSS_SEGMENT=false
C: VISUAL_GROUNDING=true,  CROSS_SEGMENT=true
```

Workload fingerprint giống nhau nhưng wall time khác cho thấy slowdown ở runtime/system; fingerprint khác cho thấy model thực sự làm thêm workload.

## Thay đổi chính trong 1.3.5.4

1. **Line-level visual grounding.** Decoder trace bổ sung `visual_coverage_gain`,
   `novel_ink_support` và `reused_attention_ratio`; guard không chỉ nhìn vị trí
   tâm attention mà đo xem token có thực sự tiêu thụ pixel mới hay chỉ đọc lại
   cùng một vùng ảnh.
2. **Global attention coordinate.** Attention của từng split segment được ánh xạ
   về tọa độ X của line crop gốc bằng `source_left/source_right`, nên có thể so
   evidence xuyên qua ranh giới segment.
3. **Cross-segment right anchor.** Tail yếu của segment trước có thể được xác
   nhận bằng head mạnh của segment kế tiếp. Đây là lớp xử lý cho hallucination
   có dạng `text thật + rác | text thật tiếp tục` mà 1.3.5.3 không thể xác nhận.
4. **Delete-only, lexically invariant.** Mọi quyết định vẫn chỉ xóa span; không
   blacklist, dictionary, LLM rewrite hay bảng sửa từ. Numeric span vẫn
   warning-only.
5. **Rejected-boundary telemetry.** Boundary không đủ chắc chắn được ghi kèm
   `failed_conditions` để lần chạy PDF thật cho biết threshold nào đang chặn
   candidate thay vì tiếp tục hạ ngưỡng mù.
6. **PP-OCRv5 Latin verifier tùy chọn.** Có thể bật recognizer thị giác độc lập
   `latin_PP-OCRv5_mobile_rec`; mặc định tắt và không được quyền rewrite. Nếu bật
   chế độ gate, auto-delete chỉ được áp dụng khi secondary recognizer nghiêng rõ
   về proposal delete-only.
7. **91 test pass, 2 integration test skip khi môi trường build thiếu VietOCR.**
   Test mới kiểm tra visual coverage/reuse, global mapping, cross-segment
   deletion, lexical invariance, numeric safety và secondary verifier conflict.

Bản đóng gói chưa chạy checkpoint Paddle/VietOCR thật. 1.3.5.4 tiếp tục ưu tiên
ngữ nghĩa hơn runtime; full coverage vẫn bật để thu đủ evidence trên hai PDF
regression trước khi chuyển corpus mới.

## Pipeline

```text
PDF upload
  → PdfDocumentSession mở một lần
  → pdf-inspector / PDFium native routing
      ├─ native page → trả text/Markdown ngay
      └─ OCR page → đưa vào recognition window
           → render 200 DPI
           → rotation / deskew / low-ink gate
           → PP-OCRv6_medium_det
           → perspective line crops
           → baseline fragment merge + occupancy column ordering 1.3.3
           → width-cap split 1.3.3 tại whitespace valley; overlap tắt
           → adaptive batches xuyên nhiều trang
           → giữ nguyên chiều rộng crop (không common-width padding)
           → VietOCR greedy trong inference_mode
           → full-coverage decoder probability + attention-to-ink evidence
           → visual coverage/reuse + global line-coordinate mapping
           → suffix/midline + cross-segment delete-only validation
           → optional multi-view fallback
           → tail validation + decoder-loop policy
           → Tesseract vie+eng kiểm chứng toàn trang
           → PP-OCRv5 batch verifier chỉ cho dòng rủi ro
           → semantic delete-only guard
           → dựng text audit, ai_safe_text và line_results
           → ghép segment theo thứ tự, ghép dòng về trang
  → merge đúng page_index
  → JSON: text + Markdown + metrics
```

Xem [PIPELINE.md](PIPELINE.md) để biết chi tiết thuật toán và metrics.

## Yêu cầu

- Windows hoặc Linux.
- Python 3.10–3.13.
- RAM khuyến nghị từ 8 GB.
- `models/vietocr/vgg_seq2seq.pth` lấy từ dự án cũ.
- Paddle model `PP-OCRv6_medium_det` cần được tải/cache khi chạy lần đầu.

## Cài đặt Windows CPU

Giải nén ZIP vào thư mục rỗng:

```powershell
$Zip = "D:\KeySoft\government-ocr-text-api-full-1.3.5.6.zip"
$Project = "D:\KeySoft\government-ocr-text-api-full"

New-Item -ItemType Directory -Path $Project -Force | Out-Null
Expand-Archive -Path $Zip -DestinationPath $Project -Force
Set-Location $Project
```

Kiểm tra project root:

```powershell
Test-Path ".\requirements.txt"
Test-Path ".\src\government_ocr_text_api\main.py"
Test-Path ".\models\vietocr\vgg_seq2seq.yml"
```

Tạo virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
python -m pip install --force-reinstall "setuptools==80.10.2"
```

Cài đúng cặp Torch/Torchvision CPU:

```powershell
python -m pip install --force-reinstall `
  "torch==2.5.1" `
  "torchvision==0.20.1" `
  --index-url https://download.pytorch.org/whl/cpu
```

Cài dependency và phục hồi đúng Pillow mà VietOCR yêu cầu:

```powershell
python -m pip install -r requirements.txt
python -m pip install --force-reinstall "Pillow==10.2.0"
python -m pip check
```

Các phiên bản quan trọng:

```text
setuptools   80.10.2
Pillow       10.2.0
torch        2.5.1+cpu
torchvision  0.20.1+cpu
vietocr      0.3.13
```

## Sao chép model VietOCR

```powershell
New-Item -ItemType Directory -Path ".\models\vietocr" -Force | Out-Null
Copy-Item `
  "D:\KeySoft\OCR-System\models\vietocr\vgg_seq2seq.pth" `
  ".\models\vietocr\vgg_seq2seq.pth" `
  -Force

Test-Path ".\models\vietocr\vgg_seq2seq.yml"
Test-Path ".\models\vietocr\vgg_seq2seq.pth"
```

Cả hai lệnh cuối phải trả `True`.

## Nâng cấp trực tiếp từ 1.3.5.1

Không cần tạo lại `.venv`, không cần cài lại dependency và không cần chép lại model.
Dừng Uvicorn, giải nén patch đè lên project root:

```powershell
Expand-Archive `
  "D:\Downloads\government-ocr-scale-invariant-grounding-patch-1.3.5.5.zip" `
  -DestinationPath "D:\KeySoft\government-ocr-text-api-full" `
  -Force

Set-Location "D:\KeySoft\government-ocr-text-api-full"
python -m compileall -q src
python -m pytest -q
```

Patch thay đổi source, test và tài liệu; không đụng vào `.venv`, `.env`, model cache
hoặc `vgg_seq2seq.pth`.

## Chạy ứng dụng

```powershell
python -m uvicorn government_ocr_text_api.main:app `
  --app-dir src `
  --host 127.0.0.1 `
  --port 8000
```

- Giao diện: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

Hoặc:

```powershell
.\scripts\run.cmd
```

## API

```http
POST /api/extract
Content-Type: multipart/form-data
```

Field `file`: PDF.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/extract" `
  -F "file=@D:\TaiLieu\van-ban.pdf;type=application/pdf"
```

Response:

```json
{
  "filename": "van-ban.pdf",
  "sha256": "...",
  "page_count": 17,
  "native_page_count": 0,
  "ocr_page_count": 17,
  "status": "complete",
  "processing_time_ms": 300000,
  "text": "===== Trang 1 =====\n...",
  "ai_safe_text": "===== Trang 1 =====\n...",
  "ai_ready": false,
  "semantic_risk_count": 2,
  "markdown": "<!-- Page 1 -->\n...",
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "ai_safe_text": "...",
      "ai_ready": false,
      "semantic_risk_count": 2,
      "line_results": []
    }
  ],
  "metrics": {},
  "warnings": ["semantic_review_required"]
}
```

`text` và `raw_text` phục vụ đối chiếu/audit, không phải đầu vào mặc định cho AI. Mỗi phần tử `line_results` có `semantic_risk`, `reasons`, `secondary_confidence` và `bbox` để truy vết.

## Cấu hình 1.3.5.8

Sao chép `.env.example` thành `.env`. Mọi biến có prefix `GOVERNMENT_OCR_`.

### Selective evidence / runtime mới

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `DECODER_EVIDENCE_FULL_COVERAGE` | `false` | Không trace mọi segment |
| `DECODER_EVIDENCE_MAX_CHECKS_PER_PAGE` | `8` | Số risk seed tối đa mỗi trang |
| `DECODER_EVIDENCE_INCLUDE_SPLIT_LINE_CONTEXT` | `true` | Trace segment cùng split line làm anchor |
| `CPU_RUNTIME_TUNING_ENABLED` | `true` | Cấu hình Torch CPU trước model load |
| `TORCH_CPU_THREADS` | `4` | Torch intra-op threads |
| `TORCH_INTEROP_THREADS` | `1` | Torch inter-op threads |
| `PADDLE_CPU_THREADS` | `4` | CPU threads cho TextDetection |
| `PADDLE_ENABLE_MKLDNN` | `false` | Backend an toàn mặc định trên Windows; nếu chủ động bật, lỗi tương thích init/inference sẽ fallback một lần |
| `OCR_NONTEXT_CROP_FILTER_ENABLED` | `true` | Chặn blank/rule crop trước VietOCR |
| `PAD_BATCHES_TO_COMMON_WIDTH` | `false` | Không thay đổi chiều rộng crop trước VietOCR |
| `SECONDARY_RECOGNIZER_ENABLED` | `true` | Bật PP-OCRv5 verifier theo batch |
| `SECONDARY_RECOGNIZER_APPLY_CHANGES` | `false` | Không cho recognizer phụ rewrite kết quả |
| `SEMANTIC_VERIFICATION_ENABLED` | `true` | Gắn mức rủi ro semantic cho từng dòng |
| `SEMANTIC_SELECTIVE_VERIFICATION_ENABLED` | `true` | Chỉ gọi PP-OCRv5 cho dòng rủi ro, có số hoặc bất đồng dấu câu sau Tesseract; tự kiểm tra toàn bộ nếu Tesseract tắt |
| `TESSERACT_VERIFIER_ENABLED` | `true` | Kiểm chứng độc lập toàn trang bằng Tesseract offline |
| `TESSERACT_FAIL_CLOSED` | `true` | Runtime lỗi, dòng không ghép hoặc confidence thấp đều buộc review |
| `TESSERACT_EXECUTABLE_PATH` | `tools/tesseract/tesseract.exe` | Runtime portable đóng gói trong dự án |
| `TESSERACT_DATA_PATH` | `tools/tesseract/tessdata_best` | Model chính xác cao `vie+eng` |
| `SEMANTIC_AUTO_TRIM_ENABLED` | `true` | Cho phép cắt suffix khi đáp ứng toàn bộ guard |
| `SEMANTIC_SECONDARY_MIN_CONFIDENCE` | `0.90` | Confidence phụ tối thiểu để auto-trim |


### Cấu hình OCR kế thừa

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `DEVICE` | `cpu` | `cpu`, `gpu:0` hoặc `cuda:0` tùy backend |
| `FALLBACK_RENDER_DPI` | `200` | DPI render trang scan |
| `LINE_DETECTION_MODEL_NAME` | `PP-OCRv6_medium_det` | Model phát hiện dòng |
| `RECOGNITION_BATCH_SIZE` | `32` | Số segment tối đa trong batch |
| `REVIEW_THRESHOLD` | `0.80` | Confidence dưới ngưỡng cần review |
| `TENSOR_CLEANUP_POLICY` | `document_end` | Cleanup tensor cuối document |

### Cấu hình tối ưu scan mới

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `RECOGNITION_WINDOW_PAGES` | `3` | Số trang OCR chuẩn bị trước mỗi window |
| `RECOGNITION_PIXEL_BUDGET` | `12288` | `max_resized_width × batch_size` tối đa |
| `RECOGNITION_BATCH_WIDTH_RATIO` | `2.0` | Chênh width tối đa trong cùng batch |
| `PAD_BATCHES_TO_COMMON_WIDTH` | `false` | Giữ nguyên chiều rộng crop; tránh thay đổi output recognizer |
| `SPLIT_WIDE_CROPS` | `true` | Cho phép chia crop dài |
| `SPLIT_WIDTH_CAPPED_CROPS` | `true` | Luôn thử chia crop sẽ bị cap ở 512 |
| `WIDE_CROP_SPLIT_THRESHOLD_RATIO` | `1.05` | Ngưỡng legacy khi tắt split theo model cap |
| `WIDE_CROP_TARGET_RATIO` | `0.92` | Mục tiêu width segment so với model max |
| `WIDE_CROP_MAX_SEGMENTS` | `6` | Giới hạn số mảnh trên mỗi crop |
| `WIDE_CROP_VALLEY_MAX_INK_RATIO` | `0.02` | Chỉ cắt tại khoảng trắng/valley đủ rõ |
| `EXPERIMENTAL_OVERLAP_SPLIT` | `false` | Chỉ bật overlap ảnh khi có benchmark ground truth xác nhận |
| `WIDE_CROP_OVERLAP_HEIGHT_RATIO` | `0.0` | Overlap nguồn quanh seam; mặc định tắt để tránh chèn token |
| `SEAM_MAX_TOKEN_OVERLAP` | `4` | Số token tối đa dùng để loại phần trùng seam |
| `SEAM_MAX_CHAR_OVERLAP` | `32` | Fallback overlap ký tự khi dấu câu làm lệch token |
| `TAIL_SEGMENT_VALIDATION` | `true` | Đánh dấu segment cuối nghi ngờ |
| `TAIL_SEGMENT_RETRY_ENABLED` | `false` | Cho phép retry thay đổi text tail; mặc định tắt |
| `TAIL_SEGMENT_MIN_CONFIDENCE` | `0.58` | Confidence dưới ngưỡng được xem là nghi ngờ khi ít mực |
| `TAIL_SEGMENT_LOW_INK_RATIO` | `0.012` | Ngưỡng mật độ mực thấp của tail segment |
| `TAIL_SEGMENT_TRAILING_BLANK_RATIO` | `0.18` | Tỷ lệ blank phải kích hoạt kiểm tra tail |
| `TAIL_SEGMENT_MAX_RETRIES_PER_PAGE` | `2` | Ngân sách retry tail mỗi trang |
| `SUPPRESS_LOW_INK_TAIL_HALLUCINATIONS` | `true` | Bỏ tail cực ít mực, confidence thấp nhưng sinh text dài |
| `HALLUCINATION_GUARD_ENABLED` | `false` | Multi-view fallback; mặc định tắt sau benchmark không tạo consensus |
| `HALLUCINATION_GUARD_APPLY_CHANGES` | `true` | Áp dụng delete-only consensus vào text cuối |
| `HALLUCINATION_GUARD_MIDLINE_ENABLED` | `false` | Không cho multi-view tự sửa insertion giữa câu |
| `HALLUCINATION_GUARD_SUFFIX_ONLY` | `true` | Chỉ cho phép delete-only ở cuối segment |
| `HALLUCINATION_GUARD_NUMERIC_APPLY_CHANGES` | `false` | Không tự động xóa span chứa chữ số |
| `HALLUCINATION_GUARD_MAX_RECHECKS_PER_PAGE` | `4` | Ngân sách recheck suffix tối đa mỗi trang |
| `HALLUCINATION_GUARD_RISK_THRESHOLD` | `0.48` | Điểm rủi ro tối thiểu để recheck |
| `HALLUCINATION_GUARD_CONFIDENCE_TOLERANCE` | `0.04` | Mức confidence retry được phép thấp hơn pass gốc |
| `HALLUCINATION_GUARD_EDGE_BLANK_RATIO` | `0.10` | Blank biên tối thiểu cho prefix/suffix deletion |
| `HALLUCINATION_GUARD_MIN_CROP_CHANGE_RATIO` | `0.06` | Mức thu hẹp crop tối thiểu để tạo view thứ hai |
| `HALLUCINATION_GUARD_NUMERIC_MIN_CROP_CHANGE_RATIO` | `0.16` | Ngưỡng nghiêm hơn khi token bị xóa chứa số |
| `HALLUCINATION_GUARD_MAX_REMOVED_TOKEN_RATIO` | `0.35` | Không xóa quá tỷ lệ token này trong một segment |
| `DECODER_EVIDENCE_ENABLED` | `true` | Đọc probability và attention trực tiếp từ decoder `vgg_seq2seq` |
| `DECODER_EVIDENCE_APPLY_CHANGES` | `true` | Cho phép delete-only span khi đủ bằng chứng ảnh–decoder |
| `DECODER_EVIDENCE_MAX_CHECKS_PER_PAGE` | `0` | `0` là không giới hạn; chỉ dùng khi tắt full coverage |
| `DECODER_EVIDENCE_TRACE_BATCH_SIZE` | `12` | Số candidate cùng kích thước trong một trace batch |
| `DECODER_EVIDENCE_CANDIDATE_SCORE_THRESHOLD` | `0.38` | Điểm risk tối thiểu để vào candidate set |
| `DECODER_EVIDENCE_MAX_DECODE_STEPS` | `128` | Giới hạn bước decode evidence |
| `DECODER_EVIDENCE_CANDIDATE_CONFIDENCE` | `0.72` | Ưu tiên kiểm tra segment có confidence dưới ngưỡng |
| `DECODER_EVIDENCE_MIN_PREFIX_CHARS` | `8` | Phần text hợp lệ tối thiểu phải giữ trước khi trim |
| `DECODER_EVIDENCE_WINDOW_TOKENS` | `6` | Cửa sổ token dùng phát hiện attention stall |
| `DECODER_EVIDENCE_MIN_UNSUPPORTED_TOKENS` | `3` | Số token cuối tối thiểu phải cùng thiếu bằng chứng |
| `DECODER_EVIDENCE_LOW_TOKEN_PROBABILITY` | `0.56` | Ngưỡng probability thấp của token nghi ngờ |
| `DECODER_EVIDENCE_LOW_INK_SUPPORT` | `0.10` | Absolute floor của ink-support threshold |
| `DECODER_EVIDENCE_RELATIVE_PROBABILITY_RATIO` | `0.68` | Tỷ lệ median probability prefix dùng tạo ngưỡng thích nghi |
| `DECODER_EVIDENCE_RELATIVE_INK_RATIO` | `0.28` | Tỷ lệ median ink support prefix dùng tạo ngưỡng thích nghi |
| `DECODER_EVIDENCE_PROBABILITY_THRESHOLD_CAP` | `0.72` | Cap ngưỡng probability thích nghi |
| `DECODER_EVIDENCE_INK_THRESHOLD_CAP` | `0.18` | Cap ngưỡng ink support thích nghi |
| `DECODER_EVIDENCE_STALL_RANGE_RATIO` | `0.035` | Biên độ attention nhỏ được xem là bị kẹt |
| `DECODER_EVIDENCE_END_MARGIN_RATIO` | `0.06` | Khoảng an toàn quanh cột mực cuối |
| `DECODER_EVIDENCE_NEAR_LOOP_SIMILARITY` | `0.74` | Độ giống tối thiểu của cửa sổ lặp biến thể |
| `DECODER_EVIDENCE_NEAR_LOOP_MIN_TOKENS` | `8` | Độ dài suffix tối thiểu để xét near-loop |
| `DECODER_EVIDENCE_NUMERIC_APPLY_CHANGES` | `false` | Span số chỉ cảnh báo, không tự động cắt |
| `DECODER_EVIDENCE_CLUSTER_WORD_EXPANSION_ENABLED` | `true` | Mở trim về đầu từ cùng attention cluster |
| `DECODER_EVIDENCE_CLUSTER_MAX_WORDS` | `2` | Số từ tối đa được mở rộng về trái |
| `DECODER_EVIDENCE_CLUSTER_ATTENTION_RANGE_RATIO` | `0.12` | Biên độ attention tối đa của cluster |
| `DECODER_EVIDENCE_CLUSTER_CENTER_DISTANCE_RATIO` | `0.08` | Khoảng cách center tối đa giữa từ trước và suffix |
| `DECODER_EVIDENCE_FULL_COVERAGE` | `true` | Trace mọi segment có text, bỏ qua giới hạn per-page |
| `DECODER_EVIDENCE_MIDLINE_ENABLED` | `true` | Phát hiện span giữa hai anchor được hỗ trợ |
| `DECODER_EVIDENCE_MIDLINE_MAX_WORDS` | `12` | Số từ tối đa của một midline span |
| `DECODER_EVIDENCE_MIDLINE_MAX_SPAN_RATIO` | `0.55` | Không xóa span chiếm tỷ lệ quá lớn của segment |
| `DECODER_EVIDENCE_EVENT_LIMIT` | `50` | Số evidence event tối đa ghi vào log mỗi trang |
| `DECODER_EVIDENCE_CROSS_SEGMENT_ENABLED` | `true` | Dùng next-segment head làm right visual anchor cho weak tail |
| `DECODER_EVIDENCE_CROSS_SEGMENT_MAX_WORDS` | `12` | Số từ cuối tối đa được xét ở một boundary |
| `DECODER_EVIDENCE_CROSS_SEGMENT_RELATIVE_SUPPORT_RATIO` | `0.68` | Span phải yếu hơn support của hai anchor theo tỷ lệ này |
| `DECODER_EVIDENCE_CROSS_SEGMENT_RELATIVE_COVERAGE_RATIO` | `0.58` | Span phải có visual coverage gain thấp hơn anchor |
| `DECODER_EVIDENCE_CROSS_SEGMENT_MIN_REUSE_RATIO` | `0.60` | Attention reuse tối thiểu để xem tail đang đọc lại vùng cũ |
| `SECONDARY_RECOGNIZER_ENABLED` | `true` | Bật PP-OCRv5 Latin verifier độc lập; có thể tải model lần đầu |
| `SECONDARY_RECOGNIZER_MODEL_NAME` | `latin_PP-OCRv5_mobile_rec` | Recognizer Latin hỗ trợ tiếng Việt dùng cho verifier |
| `SECONDARY_RECOGNIZER_APPLY_CHANGES` | `false` | Khi `true`, verifier chỉ gate delete-only proposal; không rewrite text |
| `OCR_COLUMN_AWARE_ORDERING` | `true` | Phát hiện gutter/cột cho trang scan |
| `OCR_COLUMN_OCCUPANCY_THRESHOLD` | `0.14` | Mật độ polygon tối đa trong gutter |
| `OCR_COLUMN_MIN_GAP_RATIO` | `0.03` | Bề rộng gutter tối thiểu theo chiều rộng trang |
| `OCR_SIGNATURE_BLOCK_ENABLED` | `true` | Cho phép khối chữ ký bất cân bằng |
| `OCR_SIGNATURE_BLOCK_MAX_RIGHT_LINES` | `6` | Số dòng tối đa của cột chữ ký bên phải |
| `MERGE_BASELINE_FRAGMENTS` | `true` | Gộp ký tự hẹp cùng baseline |
| `BASELINE_FRAGMENT_MAX_GAP_RATIO` | `1.0` | Khoảng cách tối đa giữa fragment và dòng theo median height |
| `BASELINE_FRAGMENT_NARROW_WIDTH_RATIO` | `1.8` | Ngưỡng xác định fragment hẹp theo median height |
| `CHAPTER_HEADING_RETRY_ENABLED` | `true` | Retry structural heading bị thiếu chỉ số |
| `CHAPTER_HEADING_RETRY_LABELS` | `chương` | Danh sách nhãn phân tách bằng dấu phẩy; regex-escape trước khi dùng |
| `CHAPTER_HEADING_EXPAND_HEIGHT_RATIO` | `4.0` | Số lần chiều cao dòng dùng để mở crop sang phải |
| `CHAPTER_HEADING_MIN_CONFIDENCE` | `0.45` | Confidence tối thiểu của tiêu đề chương retry |
| `CHAPTER_HEADING_MAX_RETRIES_PER_PAGE` | `2` | Ngân sách retry tiêu đề chương mỗi trang |
| `BEAM_RETRY_ENABLED` | `true` | Cho phép retry loop bằng beam search |
| `MAX_BEAM_RETRIES_PER_PAGE` | `1` | Ngân sách retry tối đa mỗi trang |
| `USE_TORCH_INFERENCE_MODE` | `true` | Tắt autograd trong inference |

### Preset CPU ưu tiên tốc độ

```dotenv
GOVERNMENT_OCR_RECOGNITION_WINDOW_PAGES=3
GOVERNMENT_OCR_RECOGNITION_PIXEL_BUDGET=12288
GOVERNMENT_OCR_MAX_BEAM_RETRIES_PER_PAGE=1
GOVERNMENT_OCR_PAD_BATCHES_TO_COMMON_WIDTH=false
GOVERNMENT_OCR_SPLIT_WIDE_CROPS=false
```

### Preset CPU cân bằng chất lượng

```dotenv
GOVERNMENT_OCR_RECOGNITION_WINDOW_PAGES=3
GOVERNMENT_OCR_RECOGNITION_PIXEL_BUDGET=12288
GOVERNMENT_OCR_MAX_BEAM_RETRIES_PER_PAGE=1
GOVERNMENT_OCR_PAD_BATCHES_TO_COMMON_WIDTH=false
GOVERNMENT_OCR_SPLIT_WIDE_CROPS=true
GOVERNMENT_OCR_SPLIT_WIDTH_CAPPED_CROPS=true
GOVERNMENT_OCR_WIDE_CROP_TARGET_RATIO=0.92
GOVERNMENT_OCR_EXPERIMENTAL_OVERLAP_SPLIT=false
GOVERNMENT_OCR_WIDE_CROP_OVERLAP_HEIGHT_RATIO=0.0
GOVERNMENT_OCR_TAIL_SEGMENT_VALIDATION=true
GOVERNMENT_OCR_TAIL_SEGMENT_RETRY_ENABLED=false
GOVERNMENT_OCR_TAIL_SEGMENT_MAX_RETRIES_PER_PAGE=2
GOVERNMENT_OCR_HALLUCINATION_GUARD_ENABLED=false
GOVERNMENT_OCR_HALLUCINATION_GUARD_APPLY_CHANGES=true
GOVERNMENT_OCR_HALLUCINATION_GUARD_MIDLINE_ENABLED=false
GOVERNMENT_OCR_HALLUCINATION_GUARD_SUFFIX_ONLY=true
GOVERNMENT_OCR_HALLUCINATION_GUARD_NUMERIC_APPLY_CHANGES=false
GOVERNMENT_OCR_HALLUCINATION_GUARD_MAX_RECHECKS_PER_PAGE=4
GOVERNMENT_OCR_HALLUCINATION_GUARD_RISK_THRESHOLD=0.48
GOVERNMENT_OCR_DECODER_EVIDENCE_ENABLED=true
GOVERNMENT_OCR_DECODER_EVIDENCE_APPLY_CHANGES=true
GOVERNMENT_OCR_DECODER_EVIDENCE_MAX_CHECKS_PER_PAGE=0
GOVERNMENT_OCR_DECODER_EVIDENCE_FULL_COVERAGE=true
GOVERNMENT_OCR_DECODER_EVIDENCE_MIDLINE_ENABLED=true
GOVERNMENT_OCR_DECODER_EVIDENCE_TRACE_BATCH_SIZE=12
GOVERNMENT_OCR_DECODER_EVIDENCE_LOW_TOKEN_PROBABILITY=0.56
GOVERNMENT_OCR_DECODER_EVIDENCE_LOW_INK_SUPPORT=0.10
GOVERNMENT_OCR_DECODER_EVIDENCE_NUMERIC_APPLY_CHANGES=false
GOVERNMENT_OCR_OCR_COLUMN_AWARE_ORDERING=true
GOVERNMENT_OCR_OCR_COLUMN_OCCUPANCY_THRESHOLD=0.14
GOVERNMENT_OCR_OCR_SIGNATURE_BLOCK_ENABLED=true
GOVERNMENT_OCR_OCR_SIGNATURE_BLOCK_MAX_RIGHT_LINES=6
GOVERNMENT_OCR_MERGE_BASELINE_FRAGMENTS=true
```

Không nên tăng `MAX_BEAM_RETRIES_PER_PAGE` lên 8–12 trên CPU; log trước đây cho
thấy beam retry là nguồn biến động thời gian lớn và vẫn có thể sinh decoder loop.

## Metrics mới

Trong `page_complete.metrics.layout_ordering`:

```json
{
  "layout_mode": "signature_block",
  "vertical_block_count": 3,
  "column_block_count": 1,
  "column_gutters": [1048.3],
  "column_gap_pixels": [73.4],
  "column_occupancy": [0.0476],
  "column_crossing_count": [2],
  "column_modes": ["signature_block"],
  "baseline_fragment_merge_count": 1,
  "narrow_fragment_dropped_count": 2
}
```

Trong `page_complete.metrics.recognition`:

```json
{
  "crop_count": 37,
  "segment_count": 51,
  "split_segment_count": 14,
  "split_overlap_source_pixels": 0,
  "width_cap_detected_count": 14,
  "width_cap_resolved_count": 13,
  "width_cap_unresolved_count": 1,
  "hallucination_guard_candidate_count": 0,
  "hallucination_guard_retry_count": 0,
  "hallucination_guard_consensus_count": 0,
  "hallucination_guard_removed_token_count": 0,
  "hallucination_guard_suffix_removed_count": 0,
  "hallucination_guard_midline_removed_count": 0,
  "hallucination_guard_numeric_removed_count": 0,
  "decoder_evidence_candidate_count": 18,
  "decoder_evidence_selected_count": 12,
  "decoder_evidence_unchecked_candidate_count": 6,
  "decoder_evidence_trace_count": 12,
  "decoder_evidence_trace_batch_count": 2,
  "decoder_evidence_trace_batch_size_max": 8,
  "decoder_evidence_supported_count": 12,
  "decoder_evidence_trace_mismatch_count": 0,
  "decoder_evidence_attention_stall_count": 1,
  "decoder_evidence_visual_exhausted_count": 1,
  "decoder_evidence_near_loop_count": 1,
  "decoder_evidence_trimmed_count": 1,
  "decoder_evidence_trimmed_char_count": 28,
  "decoder_evidence_suspicious_numeric_count": 1,
  "decoder_evidence_cluster_expansion_count": 1,
  "decoder_evidence_expanded_word_count": 1,
  "decoder_evidence_ms": 182.4,
  "decoder_loop_detected_count": 1,
  "decoder_loop_trimmed_count": 1,
  "empty_recognition_count": 0,
  "recognition_ms_allocated": 11300.7
}
```

`decoder_evidence_events` chứa text gốc, `proposed_text`, text thực tế sau áp dụng, span bị bỏ, offsets, loại span và lý do
(`attention_stall`, `visual_evidence_exhausted` hoặc `attention_near_loop`),
probability, attention range/progression, ink support và evidence của hai anchor. Output chỉ bị thay đổi
khi decoder trace khớp text greedy, span có đủ bằng chứng và không thuộc nhóm numeric được bảo vệ.

`hallucination_guard_events` vẫn ghi các phép delete-only được tight-crop xác nhận.
Multi-view guard vẫn suffix-only và mặc định tắt. Midline deletion chỉ thuộc decoder-evidence guard, yêu cầu hai anchor được hỗ trợ; span chứa số vẫn được giữ nguyên.

`tail_segment_suppressed` chỉ được dùng khi ảnh tail gần như không có mực,
confidence thấp nhưng model vẫn sinh chuỗi đủ dài. Các trường hợp không chắc chắn
được giữ text và gắn `tail_segment_uncertain`; pipeline không tự sửa nội dung pháp
lý bằng từ điển.

`recognition_window_ms` là thời gian thật của toàn window; `vietocr_ms` ở từng trang
là thời gian được phân bổ theo số segment trong shared batches cộng retry của chính
trang đó. Không cộng `recognition_window_ms` vào `page_total_ms` để tránh đếm trùng.

## Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -m compileall -q src
python -m pytest -q
```

Bộ test hiện có 81 test pass trong môi trường đóng gói và 2 integration test
không cần checkpoint. Hai test tự skip khi `vietocr` chưa được cài, nhưng sẽ chạy
trên môi trường ứng dụng thật có VietOCR 0.3.13. Để xác nhận hiệu
năng thực, phải benchmark trên cùng PDF, cùng CPU và cùng model cache trước/sau
patch.

## Phân tích log benchmark

Lưu terminal log vào file rồi chạy:

```powershell
python .\scripts\analyze-ocr-log.py .\logs\before.log
python .\scripts\analyze-ocr-log.py .\logs\after.log
```

So sánh `processing_time_ms`, `vietocr_ms`, `shared_batch_count`,
`segment_count`, `split_segment_count`, `tail_segment_*`, `decoder_evidence_*`,
`beam_retry_count` và `decoder_loop_trimmed_count` trên cùng PDF.

## Xác nhận native text fast path

Với PDF có text layer tốt, log phải có:

```json
{
  "event": "native_routing_complete",
  "native_page_count": 10,
  "ocr_page_count": 0,
  "pages_needing_ocr": []
}
```

Khi đó không được xuất hiện log tải Paddle/VietOCR. Tối ưu 1.3.5.4 không thay đổi nhánh
này.

## Cảnh báo có thể bỏ qua

```text
No ccache found
pkg_resources is deprecated
Multiple definitions ... for key /Info
```

Các dòng này không phải nguyên nhân request chậm. Với PDF scan, bottleneck chính là
line detection và đặc biệt là VietOCR recognition.

## Giới hạn

- Split crop dài cải thiện width-cap nhưng có thể cắt giữa một từ nếu ảnh không có
  valley khoảng trắng rõ; kết quả vẫn được đánh dấu review theo confidence.
- Document-level batching tăng peak RAM vì giữ tối đa `RECOGNITION_WINDOW_PAGES`
  trang/crop trong bộ nhớ. Mặc định 3 là lựa chọn thận trọng.
- Bản phát hành đã chạy unit test với predictor giả lập; chưa benchmark model thật
  trong môi trường đóng gói vì không kèm trọng số `.pth` và Paddle cache.

## Điều chỉnh sau benchmark 1.1.1

Bản 1.1.1 giảm split gần như về 0 ở phần lớn trang, nhưng benchmark thực tế cho
thấy 109/242 segment vẫn vượt `image_max_width`, tổng thời gian tăng từ khoảng
89,7 giây lên 109,5 giây và decoder loop dài xuất hiện lại. Bản 1.1.2:

- Hạ ngưỡng split từ `2.5` xuống `2.0 × image_max_width`.
- Vẫn chỉ cắt tại valley có mật độ mực không quá `0.025`.
- Không cưỡng bức cắt giữa chữ khi không tìm được khoảng trắng.
- Cắt chuỗi số/ký tự lặp dài và chuỗi dấu câu lặp trước khi ghép dòng.
- Bỏ loop bắt đầu ngay đầu dòng hoặc chỉ có một token nhiễu phía trước.
- Giữ ngưỡng ba lần cho token/n-gram để tránh cắt nhầm văn bản hợp lệ.

Mục tiêu của 1.1.2 là nằm giữa hai cực: không tăng segment 80–100% như 1.1.0,
nhưng cũng không để phần lớn dòng dài bị width-cap như 1.1.1.
