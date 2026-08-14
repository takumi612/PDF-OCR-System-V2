# oneDNN Runtime Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent HTTP 500 when PaddlePaddle's CPU oneDNN/PIR executor fails during text detection by rebuilding the detector without oneDNN and retrying exactly once.

**Architecture:** `PaddleLineDetector` remains the sole owner of Paddle detector lifecycle. Construction is centralized, runtime fallback recognizes only the known oneDNN/PIR compatibility exception, replaces the cached processor with a non-oneDNN processor, and reuses it for all later calls.

**Tech Stack:** Python 3.10, PaddlePaddle 3.3.1, PaddleOCR 3.7.0, PaddleX 3.7.2, pytest, Ruff.

## Global Constraints

- Keep oneDNN enabled when requested; fall back only for the known `ConvertPirAttribute2RuntimeAttribute` oneDNN failure.
- Retry the same image exactly once with `enable_mkldnn=False`.
- Persist and reuse the fallback processor for later pages and requests.
- Propagate unrelated exceptions and any exception from the fallback retry unchanged.
- Do not change HTTP schemas, native PDF routing, VietOCR behavior, or dependency versions.
- The target directory has no `.git`; no commit steps are possible.

---

### Task 1: Runtime fallback regression behavior

**Files:**
- Modify: `tests/test_line_detector.py`
- Modify: `src/government_ocr_text_api/line_detector.py:650-760`

**Interfaces:**
- Consumes: `PaddleLineDetector.detect(page: PageImage) -> list[LinePolygon]` and `paddleocr.TextDetection(**kwargs)`.
- Produces: `_create_processor(enable_mkldnn: bool) -> Any`, `_is_onednn_pir_compatibility_error(exc: Exception) -> bool`, and persistent runtime fallback state in `init_metrics`.

- [ ] **Step 1: Write the failing known-error test**

Add a fake `TextDetection` constructor through `monkeypatch` that records constructor arguments. Its oneDNN instance raises:

```python
NotImplementedError(
    "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
    "[pir::ArrayAttribute<pir::DoubleAttribute>] "
    "(at onednn_instruction.cc:118)"
)
```

Its non-oneDNN instance returns one valid `dt_polys`/`dt_scores` payload. Call `detector.detect(page)` twice and assert the real detector behavior:

```python
assert len(first) == 1
assert len(second) == 1
assert [call["enable_mkldnn"] for call in constructor_calls] == [True, False]
assert detector.init_metrics["fallback_used"] is True
assert detector.init_metrics["fallback_phase"] == "predict"
assert detector.init_metrics["mkldnn_effective"] is False
```

This catches removal of retry, wrong fallback arguments, or failure to reuse the safe processor.

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_line_detector.py::test_detector_falls_back_after_known_onednn_predict_failure_and_reuses_processor
```

Expected: FAIL because `detect()` currently propagates the `NotImplementedError`.

- [ ] **Step 3: Write the unrelated-error test**

Use a fake internally constructed processor whose `predict()` raises `RuntimeError("unrelated detector failure")`. Assert that `detect()` raises the same exception and that only the oneDNN processor was constructed:

```python
with pytest.raises(RuntimeError, match="unrelated detector failure"):
    detector.detect(page)
assert [call["enable_mkldnn"] for call in constructor_calls] == [True]
```

This catches overly broad exception swallowing.

- [ ] **Step 4: Run both tests to verify RED**

Run the two named tests. Expected: the known-error test fails at inference; the unrelated-error test passes because current code already propagates it.

- [ ] **Step 5: Implement minimal detector lifecycle and retry**

In `PaddleLineDetector`:

1. Add `fallback_phase: None` to `init_metrics`.
2. Extract TextDetection argument construction into `_create_processor(enable_mkldnn: bool)` while retaining the existing `cpu_threads` compatibility retry.
3. Preserve constructor fallback and set `fallback_phase="init"` when it is used.
4. Add a strict classifier requiring the exception type to be `NotImplementedError`, the message to contain `ConvertPirAttribute2RuntimeAttribute`, and the message to contain `onednn`.
5. Wrap `processor.predict(image)` in `detect()`. On an eligible failure, create a processor with `enable_mkldnn=False`, update metrics, replace `self.processor`, and retry once.
6. Do not wrap the fallback retry in another fallback loop.

- [ ] **Step 6: Run detector tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_line_detector.py
```

Expected: all detector tests pass.

---

### Task 2: Integration verification and documentation alignment

**Files:**
- Modify: `README.md` only if the existing fallback description needs clarification.
- Verify: `src/government_ocr_text_api/line_detector.py`
- Verify: `tests/test_line_detector.py`

**Interfaces:**
- Consumes: effective settings `paddle_enable_mkldnn=True`, `paddle_mkldnn_fallback=True` and cached `PP-OCRv6_medium_det`.
- Produces: verified CPU detector behavior and accurate operational documentation.

- [ ] **Step 1: Run focused unit tests**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_line_detector.py tests\test_ocr_pipeline.py tests\test_extractor_routing.py tests\test_api.py
```

Expected: zero failures.

- [ ] **Step 2: Run Ruff**

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
```

Expected: exit code 0.

- [ ] **Step 3: Run the full unit suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: zero failures. If the known long-running VietOCR integration tests exceed the limit, report the exact timeout and retain the focused-suite result separately.

- [ ] **Step 4: Run a real Paddle detector smoke test**

Start a fresh Python process, instantiate `PaddleLineDetector(Settings())`, render an in-memory page containing `TEST OCR 123`, and call `detect()` twice. Assert:

```python
detector.init_metrics["fallback_used"] is True
detector.init_metrics["fallback_phase"] == "predict"
detector.init_metrics["mkldnn_effective"] is False
```

Expected: both calls complete; the log may show the oneDNN error only on the first call.

- [ ] **Step 5: Run an isolated API integration test**

Launch a second Uvicorn instance on an unused port with the patched source, submit a small scanned PDF fixture, and assert that `/api/extract` does not return HTTP 500. Do not restart or alter the user's existing PID 10828 during verification.

- [ ] **Step 6: Align README wording**

If README still describes fallback as initialization-only or ambiguously, update it to state that known oneDNN/PIR failures during the first inference are retried once without oneDNN and the safe predictor is reused.

- [ ] **Step 7: Inspect the final diff**

Confirm that changes are limited to detector lifecycle, regression tests, plan/spec documentation, and the narrow README clarification. Confirm no environment, model, or dependency files changed.
