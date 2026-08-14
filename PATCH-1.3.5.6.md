# Patch 1.3.5.5 → 1.3.5.6

Patch này chỉ chứa source/test/tài liệu thay đổi. Không chứa `.venv`, `.env`, `*.pth`, Paddle model cache hoặc runtime data.

## Cài trực tiếp

```powershell
Set-Location "D:\KeySoft\government-ocr-text-api-full"
# Dừng Uvicorn trước.
Expand-Archive `
  "D:\Downloads\government-ocr-selective-runtime-patch-1.3.5.6.zip" `
  -DestinationPath "." `
  -Force

& ".\.venv\Scripts\Activate.ps1"
$env:PYTHONPATH = "$PWD\src"
python -m compileall -q src scripts tests
python -m pytest -q
```

## Kiểm tra version/config

```powershell
python -c "
import government_ocr_text_api
from government_ocr_text_api.config import Settings
s = Settings()
print('package:', government_ocr_text_api.__version__)
print('app:', s.app_version)
print('full_coverage:', s.decoder_evidence_full_coverage)
print('max_checks:', s.decoder_evidence_max_checks_per_page)
print('split_context:', s.decoder_evidence_include_split_line_context)
print('torch_threads:', s.torch_cpu_threads)
print('paddle_threads:', s.paddle_cpu_threads)
print('paddle_mkldnn:', s.paddle_enable_mkldnn)
print('nontext_filter:', s.ocr_nontext_crop_filter_enabled)
print('secondary:', s.secondary_recognizer_enabled)
"
```

Kỳ vọng mặc định:

```text
package: 1.3.5.6
app: 1.3.5.6
full_coverage: False
max_checks: 8
split_context: True
torch_threads: 4
paddle_threads: 4
paddle_mkldnn: True
nontext_filter: True
secondary: False
```

## A/B nhanh nếu cần rollback hành vi mới

Không cần downgrade source. Có thể tắt từng tối ưu qua `.env`:

```text
# Quay evidence về kiểu 1.3.5.5
GOVERNMENT_OCR_DECODER_EVIDENCE_FULL_COVERAGE=true
GOVERNMENT_OCR_DECODER_EVIDENCE_MAX_CHECKS_PER_PAGE=0

# Tắt CPU tuning
GOVERNMENT_OCR_CPU_RUNTIME_TUNING_ENABLED=false

# Tắt oneDNN/MKL-DNN detector
GOVERNMENT_OCR_PADDLE_ENABLE_MKLDNN=false

# Tắt crop filter
GOVERNMENT_OCR_OCR_NONTEXT_CROP_FILTER_ENABLED=false
```

## Metrics cần gửi lại sau khi chạy PDF thật

```text
processing_time_ms
line_detection_ms
vietocr_ms
recognition_window_ms

greedy_batch_fallback_count
greedy_batch_fallback_segment_count

decoder_evidence_candidate_count
decoder_evidence_seed_selected_count
decoder_evidence_context_forced_count
decoder_evidence_selected_count
decoder_evidence_trace_count
decoder_evidence_trace_batch_count
decoder_evidence_ms

decoder_evidence_cross_segment_candidate_count
decoder_evidence_cross_segment_trimmed_count
decoder_evidence_cross_segment_rejected_count

nontext_crop_filter
detector_runtime
performance_diagnostics.runtime_start
performance_diagnostics.runtime_end
```
