# Semantic-Safe OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover legal lines removed by the current non-text filter, remove only proven unsupported OCR suffixes, and return a masked AI-safe text channel with line-level semantic evidence.

**Architecture:** Keep VietOCR as the primary source of returned characters, run PP-OCRv5 Latin as a batched independent verifier, and place comparison-only logic in a focused `semantic_guard.py` module. The OCR pipeline preserves the auditable primary text while producing `ai_safe_text` that replaces high-risk lines with structured placeholders.

**Tech Stack:** Python 3.10, FastAPI/Pydantic, Pillow/OpenCV/NumPy, VietOCR/PyTorch, PaddleOCR/PaddleX, pytest.

## Global Constraints

- Never use dictionary, blacklist, corpus-specific phrase replacement, or LLM rewriting.
- Never auto-delete a span containing a digit.
- Automatic correction is deletion-only and suffix-only.
- Secondary text may validate or flag primary text but may never replace it.
- Preserve the existing `text` response field and add backward-compatible optional/defaulted fields.
- The source directory has no `.git`; verification is mandatory, but commit steps are not available in this checkout.

---

### Task 1: Restore text coverage and deterministic line inputs

**Files:**
- Modify: `src/government_ocr_text_api/config.py`
- Modify: `src/government_ocr_text_api/ocr_pipeline.py`
- Test: `tests/test_ocr_pipeline.py`
- Test: `tests/test_vietocr_batch.py`

**Interfaces:**
- Consumes: existing `_nontext_crop_reason(crop: LineCrop, settings: Settings) -> str | None`.
- Produces: conservative rule filtering and `Settings().pad_batches_to_common_width is False`.

- [ ] **Step 1: Add a failing realistic glyph-band regression test**

```python
def test_nontext_filter_keeps_padded_component_rich_legal_text():
    image = Image.new("RGB", (720, 52), "white")
    pixels = image.load()
    for x in range(20, 700, 30):
        for px in range(x, min(x + 20, 710)):
            for y in range(17, 36):
                pixels[px, y] = (0, 0, 0)
    crop = _line_crop("p0000-l0000", 0, 0, 720, 52, image)
    assert _nontext_crop_reason(crop, Settings()) is None
```

- [ ] **Step 2: Run the focused filter tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ocr_pipeline.py -k nontext -q`

Expected: the new glyph-band test fails with `horizontal_rule_or_dotted_leader`.

- [ ] **Step 3: Restrict filtering to near-blank or proven continuous rules**

Replace the broad `thin_band or solid_rule or dotted_leader` decision with a continuous-rule decision requiring all of:

```python
proven_continuous_rule = (
    ink_ratio <= settings.ocr_nontext_rule_max_ink_ratio
    and active_row_ratio <= 0.15
    and max_component_width_ratio >= 0.80
    and max_component_height_ratio <= 0.20
)
if proven_continuous_rule:
    return "horizontal_rule_or_dotted_leader"
```

Keep the near-blank branch unchanged. Do not inspect recognized words.

- [ ] **Step 4: Add a failing default-padding test**

```python
def test_common_width_padding_is_disabled_by_default():
    assert Settings().pad_batches_to_common_width is False
```

- [ ] **Step 5: Run the default-padding test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vietocr_batch.py -k common_width_padding_is_disabled_by_default -q`

Expected: FAIL because the current default is `True`.

- [ ] **Step 6: Change only the default to false and verify GREEN**

Change `pad_batches_to_common_width: bool = False`, then run:

`.venv\Scripts\python.exe -m pytest tests/test_ocr_pipeline.py tests/test_vietocr_batch.py -q`

Expected: all tests pass.

### Task 2: Build the deletion-only semantic decision module

**Files:**
- Create: `src/government_ocr_text_api/semantic_guard.py`
- Create: `tests/test_semantic_guard.py`
- Modify: `src/government_ocr_text_api/config.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: primary text/confidence/error and optional secondary text/confidence.
- Produces: `SemanticDecision`, `evaluate_semantic_line`, and new semantic threshold settings.

- [ ] **Step 1: Write failing tests for the approved decision policy**

```python
def test_removes_only_unsupported_non_numeric_suffix():
    decision = evaluate_semantic_line(
        primary_text=("59/2015/TT-BCT Ngày 31 tháng 12 năm 2015 của Bộ Công "
                      "Thương quy định về người thuy nhi viện thuy nghiệp thuyên thiến"),
        primary_confidence=0.47,
        primary_error_code="decoder_loop_trimmed",
        secondary_text=("59/2015/TT-BCT ngày 31 tháng 12 năm 2015 ca B Công "
                        "Thưng quy đnh v"),
        secondary_confidence=0.97,
        settings=Settings(),
    )
    assert decision.text.endswith("quy định về")
    assert decision.raw_text.endswith("thuyên thiến")
    assert decision.reasons == ("unsupported_suffix_removed",)
    assert decision.risk == "medium"


def test_numeric_suffix_is_never_deleted():
    decision = evaluate_semantic_line(
        primary_text="Điều khoản có hiệu lực năm 2025 2026 2027",
        primary_confidence=0.40,
        primary_error_code="decoder_loop_trimmed",
        secondary_text="Điều khoản có hiệu lực năm",
        secondary_confidence=0.99,
        settings=Settings(),
    )
    assert decision.text.endswith("2025 2026 2027")
    assert decision.risk == "high"
    assert "numeric_suffix_protected" in decision.reasons


def test_secondary_omission_disagreement_is_masked_not_rewritten():
    decision = evaluate_semantic_line(
        primary_text="xâm phạm ninh quốc gia, khủng bố",
        primary_confidence=0.85,
        primary_error_code=None,
        secondary_text="xâm phm an ninh quc gia, khng b",
        secondary_confidence=0.98,
        settings=Settings(),
    )
    assert decision.text == "xâm phạm ninh quốc gia, khủng bố"
    assert decision.risk == "high"
    assert "secondary_indicates_primary_omission" in decision.reasons
```

- [ ] **Step 2: Run the semantic tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_semantic_guard.py -q`

Expected: import failure because `semantic_guard.py` does not exist.

- [ ] **Step 3: Implement comparison-only normalization and immutable decision**

Create:

```python
@dataclass(frozen=True)
class SemanticDecision:
    text: str
    raw_text: str | None
    risk: Literal["none", "medium", "high"]
    reasons: tuple[str, ...]
    secondary_confidence: float | None


def evaluate_semantic_line(
    *,
    primary_text: str,
    primary_confidence: float,
    primary_error_code: str | None,
    secondary_text: str | None,
    secondary_confidence: float | None,
    settings: Settings,
) -> SemanticDecision:
    primary_tokens = _lexical_tokens(primary_text)
    if not primary_text or not primary_tokens:
        return SemanticDecision(
            primary_text, None, "high", ("empty_primary_text",), secondary_confidence
        )
    if not secondary_text or secondary_confidence is None:
        risky_primary = bool(
            primary_error_code
            or primary_confidence < settings.semantic_primary_low_confidence
        )
        return SemanticDecision(
            primary_text,
            None,
            "high" if risky_primary else "none",
            ("secondary_unavailable",) if risky_primary else (),
            None,
        )
    comparison = _compare_token_sequences(primary_text, secondary_text)
    return _decision_from_comparison(
        primary_text=primary_text,
        primary_confidence=primary_confidence,
        primary_error_code=primary_error_code,
        secondary_confidence=secondary_confidence,
        comparison=comparison,
        settings=settings,
    )
```

Implement `_lexical_tokens`, `_compare_token_sequences`, and `_decision_from_comparison` in the same module. Use Unicode token spans, comparison-only accent stripping, positional fuzzy similarity, and the exact thresholds in the approved design. Slice the original primary string at its token boundary; never return normalized or secondary characters.

Add exact settings defaults:

```python
secondary_recognizer_enabled: bool = True
semantic_verification_enabled: bool = True
semantic_auto_trim_enabled: bool = True
semantic_secondary_min_confidence: float = 0.90
semantic_primary_low_confidence: float = 0.62
semantic_suffix_min_extra_tokens: int = 3
semantic_prefix_min_tokens: int = 6
semantic_position_match_ratio: float = 0.72
semantic_tail_match_ratio: float = 0.75
```

Add an autouse test fixture that sets `GOVERNMENT_OCR_SECONDARY_RECOGNIZER_ENABLED=false`; tests that exercise the verifier must pass `secondary_recognizer_enabled=True` explicitly with a fake secondary model. This prevents unit tests from loading network/cache models while keeping production verification enabled by default.

- [ ] **Step 4: Add boundary tests**

Cover secondary unavailable, low secondary confidence, fewer than three extra tokens, fewer than six anchors, tail-match below 0.75, and empty primary recognition. Each case asserts exact risk and reasons.

- [ ] **Step 5: Run semantic tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_semantic_guard.py -q`

Expected: all tests pass.

### Task 3: Batch the independent recognizer and attach semantic evidence

**Files:**
- Modify: `src/government_ocr_text_api/models.py`
- Modify: `src/government_ocr_text_api/vietocr_recognizer.py`
- Test: `tests/test_vietocr_batch.py`

**Interfaces:**
- Consumes: `evaluate_semantic_line` from Task 2.
- Produces: `VietOcrRecognizer._secondary_recognize_lines(images)`, enriched `Recognition`, and semantic verification metrics.

- [ ] **Step 1: Write a failing batched-secondary test**

Use a fake Paddle result exposing `json={"res": {"rec_text": "verified prefix", "rec_score": 0.97}}` and assert one `predict` call receives all crop images with batch size 32. Assert the known suffix is trimmed in the returned `Recognition`, while `raw_text` and semantic metadata remain present.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vietocr_batch.py -k semantic_secondary_batch -q`

Expected: FAIL because no batch method or semantic fields exist.

- [ ] **Step 3: Extend `Recognition` compatibly**

Append defaulted fields:

```python
raw_text: str | None = None
semantic_risk: Literal["none", "medium", "high"] = "none"
semantic_reasons: tuple[str, ...] = ()
secondary_confidence: float | None = None
```

- [ ] **Step 4: Implement one batched verifier pass after primary guards**

Create `_secondary_recognize_lines(images)` using the cached model and parse every result. Apply semantic decisions to the assembled per-crop `Recognition` list. Add metrics for verified, unavailable, high-risk, and auto-trimmed counts.

Initialize `TextRecognition` with `device`, `cpu_threads`, and the configured oneDNN preference. On the exact known Paddle PIR/oneDNN `NotImplementedError`, rebuild with `enable_mkldnn=False`, retry once, and reuse the safe instance.

- [ ] **Step 5: Add failure-isolation tests**

Assert model unavailable or predict failure does not fail the document. It must preserve primary text, set `secondary_unavailable`, and mark already risky primary lines high-risk.

- [ ] **Step 6: Run recognizer tests and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vietocr_batch.py -q`

Expected: all tests pass.

### Task 4: Expose the AI-safe response contract

**Files:**
- Modify: `src/government_ocr_text_api/models.py`
- Modify: `src/government_ocr_text_api/ocr_pipeline.py`
- Modify: `src/government_ocr_text_api/extractor.py`
- Test: `tests/test_ocr_pipeline.py`
- Test: `tests/test_extractor_routing.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: enriched `Recognition` from Task 3.
- Produces: `OcrLineResult`, page/document `ai_safe_text`, `ai_ready`, and risk count.

- [ ] **Step 1: Write failing page-finalization tests**

Assert a high-risk line remains in `PageResult.text` and `line_results`, while `PageResult.ai_safe_text` contains only:

```text
[OCR_SEMANTIC_RISK page=1 line=2 reasons=secondary_indicates_primary_omission]
```

Assert a safe line remains verbatim and a medium-risk auto-trim line uses validated text.

- [ ] **Step 2: Run the focused pipeline tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ocr_pipeline.py -k ai_safe -q`

Expected: FAIL because the new response fields do not exist.

- [ ] **Step 3: Add response models**

Add `OcrLineResult` with line index, crop id, text/raw text, confidence, error code, semantic risk/reasons, secondary confidence, and `[x0, y0, x1, y1]` bbox. Add defaulted page and document fields so existing constructors stay valid.

- [ ] **Step 4: Build both page channels in `_finalize`**

Keep geometry deduplication behavior. For every crop, create a line result. Append validated text to `text`; append either validated text or a structured placeholder to `ai_safe_text`. Empty failed recognition also emits a placeholder instead of disappearing silently.

- [ ] **Step 5: Write and verify failing document aggregation tests**

Assert page markers are present in both document channels, `ai_ready` is false if any page has high risk, and `semantic_risk_count` is the sum of high-risk lines.

- [ ] **Step 6: Aggregate the AI-safe channel and verify GREEN**

Update native pages with `ai_safe_text=text` and `ai_ready=True`. Add a warning when semantic review is required. Run:

`.venv\Scripts\python.exe -m pytest tests/test_ocr_pipeline.py tests/test_extractor_routing.py tests/test_api.py -q`

Expected: all tests pass.

### Task 5: Document, verify, and run both regression PDFs

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Create: response artifacts under `D:\KeySoft\OCR-System\ocr-results\semantic-safe-2026-08-09\`

**Interfaces:**
- Consumes: complete semantic-safe API.
- Produces: operational documentation, fresh responses, semantic audit summary.

- [ ] **Step 1: Update configuration and consumer documentation**

Document the secondary model cache/download, no-padding default, semantic thresholds, exact placeholder syntax, and the rule: recommendation systems must consume `ai_safe_text`, check `ai_ready`, and inspect `line_results` when false.

- [ ] **Step 2: Run the complete non-integration suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Start one isolated Uvicorn process and submit both PDFs**

Save complete JSON responses and server logs. Do not reuse an unrelated user process.

- [ ] **Step 4: Verify exact semantic regressions**

Programmatically assert:

- `30-ttg` page 1 contains the recovered title subject line and recovered Article 1/2/3 continuation lines;
- `01-bct` does not contain the known unsupported suffix in `text` or `ai_safe_text`;
- both responses include `line_results`, `ai_safe_text`, `ai_ready`, and risk counts;
- neither response returns HTTP 500 or the Paddle PIR/oneDNN error.

- [ ] **Step 5: Render/source-audit remaining high-risk lines**

Compare all high-risk page/line coordinates against the source PDF. Record which are omissions, stamp/signature noise, numeric uncertainty, or recognizer disagreement. Do not claim perfect OCR if any remain; the completion criterion is that unverified lexical content is absent from `ai_safe_text` and explicitly reported.

- [ ] **Step 6: Run fresh final verification**

Run the full test suite again and parse both saved responses in one command. Record test count, HTTP status, latency, recovered-line count, auto-trim count, and high-risk count.
