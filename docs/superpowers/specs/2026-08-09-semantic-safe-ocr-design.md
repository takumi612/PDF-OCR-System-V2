# Semantic-Safe OCR Design

**Date:** 2026-08-09

## Goal

Recover legal lines currently removed by the pre-recognition filter, suppress only
visually unsupported OCR suffix insertions, and expose an AI-safe document contract
that prevents uncertain text from being treated as verified facts.

## Evidence and root causes

- Both regression PDFs complete with HTTP 200 after the Paddle oneDNN fallback,
  therefore the remaining problem is semantic quality rather than server stability.
- On page 1 of `30-ttg.signed.pdf`, every one of the seven crops classified as
  `horizontal_rule_or_dotted_leader` is real text. The removed crops contain the
  subjects and objects missing from the title and Articles 1-3.
- Padding line images to a shared batch width changes VietOCR output and introduced
  an unsupported suffix in a legal sentence. Disabling shared-width padding restored
  one affected sentence without reducing batch size.
- Full decoder-evidence coverage and the tight-crop guard did not reliably remove
  the remaining hallucination and imposed disproportionate latency.
- `latin_PP-OCRv5_mobile_rec` reads the hallucinated crop with confidence above
  0.97 and stops at the last visible word. Across all 16 pages, a conservative
  accent-insensitive prefix comparison found one non-numeric suffix candidate, and
  it is the known unsupported `người thuy...` insertion.
- The secondary recognizer can also expose omissions, but its Vietnamese diacritics
  are not accurate enough to replace primary text.

## Safety invariants

1. Never use dictionary, blacklist, corpus-specific phrase replacement, or LLM
   rewriting inside the OCR pipeline.
2. Never auto-delete a candidate span containing a digit.
3. Auto-correction is deletion-only and suffix-only. Existing primary characters
   before the verified boundary are preserved byte-for-byte.
4. Missing or conflicting text is never reconstructed from the secondary model.
   It is marked high-risk and masked from `ai_safe_text`.
5. `text` remains backward compatible and auditable; `raw_text` remains available
   for every changed line.
6. Native PDF text remains AI-safe unless the existing native router requires OCR.

## Pipeline

```text
PDF page
  -> PP-OCRv6 detection with oneDNN fallback
  -> conservative non-text filter
       - remove near-blank crop
       - remove only a proven thin continuous rule
       - keep dotted/component-rich crops
  -> line crops / safe width split
  -> VietOCR batches without synthetic common-width padding
  -> existing decoder evidence and loop guards
  -> batched PP-OCRv5 Latin verification for every OCR line
  -> semantic decision
       - strict non-numeric suffix consensus: delete suffix only
       - disagreement/omission: keep raw primary and mark high risk
       - agreement: keep primary
  -> PageResult.text + PageResult.ai_safe_text + line_results
  -> document text + document ai_safe_text + ai_ready
```

## Semantic decision

Comparison tokenizes Unicode words and identifiers, case-folds them, converts
Vietnamese diacritics to a comparison-only ASCII form, and performs fuzzy positional
matching. Comparison normalization never changes returned OCR text.

An automatic suffix trim requires all conditions:

- semantic verification and auto-trim are enabled;
- secondary confidence is at least `0.90`;
- primary recognition already has an error code or confidence below `0.62`;
- both outputs have at least six comparable tokens;
- primary has at least three extra trailing tokens;
- at least 72% of aligned positions match and at least 75% of the final four
  secondary positions match;
- the removed suffix contains no digit.

If accepted, only the primary suffix is deleted. The line receives reason
`unsupported_suffix_removed`, keeps `raw_text`, and remains medium-risk for audit.

A line is high-risk when it is empty/failed, has an unresolved width/tail/heading
condition, or a high-confidence secondary reading indicates a primary omission or
material disagreement. High-risk text stays in `text` and `line_results`, but its
`ai_safe_text` representation is an explicit placeholder.

## API contract

`PageResult` adds:

- `ai_safe_text: str`
- `ai_ready: bool`
- `line_results: list[OcrLineResult]`

`OcrLineResult` contains:

- page/line index and crop id;
- validated `text` and optional `raw_text`;
- confidence and existing error code;
- `semantic_risk`: `none`, `medium`, or `high`;
- machine-readable `semantic_reasons`;
- secondary confidence (secondary text stays diagnostic and is not copied into
  `ai_safe_text`);
- source bounding box.

`ExtractResponse` adds:

- `ai_safe_text: str`
- `ai_ready: bool`
- `semantic_risk_count: int`

For a high-risk OCR line, `ai_safe_text` contains:

```text
[OCR_SEMANTIC_RISK page=1 line=12 reasons=secondary_omission]
```

The untrusted lexical content is deliberately absent from the AI-safe channel.

## Acceptance criteria

1. Existing API and unit tests remain compatible.
2. The realistic padded glyph-band regression crop is not classified as a rule;
   a true continuous horizontal rule is still removed.
3. Default recognition does not synthesize shared-width right padding.
4. The known `người thuy...` suffix is removed without replacing the preceding
   primary text.
5. Numeric suffix candidates are retained and marked high-risk.
6. Primary/secondary omission disagreement creates a high-risk line and an
   `OCR_SEMANTIC_RISK` placeholder.
7. Both regression PDFs return HTTP 200 with complete saved responses.
8. `30-ttg` output recovers the previously filtered subject/object lines on page 1.
9. `01-bct` output no longer exposes the known unsupported suffix in `text` or
   `ai_safe_text`.
10. The final handoff reports latency, risk counts, recovered lines, remaining
    high-risk pages, and explicitly instructs the recommendation system to consume
    `ai_safe_text` rather than `text`.
