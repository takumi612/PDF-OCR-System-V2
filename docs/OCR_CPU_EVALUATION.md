# OCR CPU evaluation for Vietnamese legal scans

Date: 2026-08-09

## Safety objective

The API keeps the primary VietOCR text unless a conservative correction has
independent OCR support or matches a narrow, unambiguous Vietnamese legal
context. Numeric/legal-code disagreement always rejects a rewrite. Every applied
change keeps `raw_text` and a structured reason. A remaining high-confidence
disagreement marks the line high risk and removes it from `ai_safe_text`.

No OCR engine can guarantee semantic correctness. The production guarantee is
therefore procedural: uncertain text is surfaced for human comparison with the
source PDF instead of being presented as AI-ready text.

## Precision-first 5-document benchmark (1.3.5.9)

The fixed benchmark contains five scan PDFs, 19 pages and 710 detected lines.
It was repeated after every behavioral change on the same CPU/offline host.

| Round | Seconds | Semantic-risk flags | Retry attempted | Retry applied | Outcome |
|---:|---:|---:|---:|---:|---|
| Baseline | 573.974 | 410 | 0 | 0 | 5 known meaning-changing errors |
| 1 | 652.322 | 384 | 65 | 26 | rejected: whole-line rewrites damaged diacritics |
| 2 | 684.338 | 399 | 25 | 11 | safe token fusion, 3 known errors remained |
| 3 | 879.335 | 395 | 25 | 15 | axis-aligned retry, 4/5 known errors fixed |
| 4 | 894.068 | 389 | 65 | 21 | 5/5 fixed; punctuation regressions found |
| 5 | 663.133 | 389 | 65 | 21 | punctuation/case preserved; fake trailing quote found |
| 6 | 704.798 | 364 | 132 | 46 | quote blocked; diacritic and terminal-comma regressions found |
| 7 | 745.037 | 369 | 180 | 41 | all eligible lines retried; list spacing still inconsistent |
| 8 | 740.506 | 348 | 180 | 72 | surface repair added; duplicate punctuation found |
| 9 | 746.432 | 351 | 180 | 66 | duplicate punctuation blocked; verified quote regressed |
| 10 | 721.462 | 366 | 180 | 45 | three-engine punctuation and narrow spelling repair added |
| 11 | 731.922 | 366 | 180 | 45 | accepted under semantic/spelling criterion |

Round 11 also applied 148 three-engine separator decisions. The higher risk
count than round 8 is not lower OCR quality: every automatically changed line
is deliberately review-flagged. The final acceptance checks were:

- all 5 known meaning-changing lines exactly matched their manually reviewed text;
- 0 hits across 710 final `line_results.text` values for the detected spelling/
  hallucination patterns (`bồ/bố/bỗ sung`, `rủ ro`, `cơ cở`, `hợp đồng năm giữ`,
  `Uu tiên`, `phân bự nguồn lực`, `nguy nguy nguy`, and related contexts);
- all 180 retry-eligible lines were attempted;
- no Paddle/oneDNN runtime failure across the repeated corpus runs.

This benchmark does not claim absolute CER/WER because all 710 lines have not
been independently transcribed as ground truth. The validated conclusion is
that the known semantic failures were corrected and the detected recurring
spelling/hallucination classes were absent from the final sample.

## Local 27-page corpus

| Document | Pages | Type |
|---|---:|---|
| `10camau.signed.pdf` | 6 | scan-only government decision |
| `276-ndcp.signed.pdf` | 13 | scan-only decree |
| `08-qh.signed.pdf` | 8 | scan-only law |

Baseline VietOCR + Paddle detection processed all 27 pages in 487.5 seconds
(about 18.1 seconds/page) and already marked 217 lines as high semantic risk.

The page-level Tesseract replay matched 884 of 923 detected lines (95.8%) in
63.3 seconds. It found 416 high-confidence disagreements: 197 diacritic, 76
numeric, and 143 material-text disagreements. A further 39 unmatched and 12
low-confidence lines are fail-closed for human review.

PP-OCRv6 medium completed all 27 pages in 1,891.1 inference seconds (70.0
seconds/page) with mean reported confidence 0.9733. That confidence was not
calibrated for this corpus: visible errors included `Độc lập` → `Đc lâp`,
`ỦY` → `ÚY`, and `ngày 14` → `ngày 44`. It is therefore unsuitable as an
authoritative replacement on this CPU host, although it remains useful as a
third benchmark signal.

These counts are safety-gate coverage, not CER/WER. CER/WER must not be reported
until every reference line has been manually transcribed or verified against an
authoritative digital source.

## Selective legal-safe verifier (1.3.5.8)

PP-OCRv5 now runs after the page-wide Tesseract pass and only for lines that
meet at least one conservative routing condition: primary error/low confidence,
Tesseract semantic risk, any digit/legal identifier, or a punctuation-signature
disagreement. If Tesseract is disabled or unavailable in fail-closed mode, the
route retains full verification coverage. Candidates are flattened across the
whole OCR page window before one PP-OCRv5 batch call, avoiding the per-page
batch regression found during the first implementation trial.

Final 27-page comparison against the previous Tesseract + full-PP pipeline:

| Metric | Full PP | Selective legal-safe window batch | Change |
|---|---:|---:|---:|
| Detected lines | 923 | 923 | 0 |
| Lines sent to PP-OCRv5 | 923 | 724 | -199 (-21.6%) |
| PP-OCRv5 measured time | 425.9 s | 331.4 s | -94.5 s (-22.2%) |
| End-to-end time | 850.2 s | 743.9 s | -106.3 s (-12.5%) |
| High-risk lines | 502 | 503 | +1 |
| Previously high-risk lines lost | 0 | 0 | none |
| Exact primary text changes | 0 | 0 of 923 | none |

Per document, the final run took 181.4 seconds for `10camau` (152/198 PP
candidates), 341.3 seconds for `276-ndcp` (345/458 candidates), and 221.2
seconds for `08-qh` (227/267 candidates). The exact comparison artifact is
`ocr-results/precision-cpu-2026-08-09/selective-legal-safe-window-batch/comparison-full-vs-selective.json`
in the evaluation workspace.

The returned OCR text is byte-for-byte unchanged on all 923 aligned lines both
against the full-PP run and against the original 27-page primary baseline.
Therefore this change does **not** claim a CER/WER accuracy increase: measured
text-output change is 0%. Its verified gains are lower secondary-OCR workload,
12.5% lower end-to-end time in this run, and no reduction in safety coverage.
Compared with the pre-Tesseract baseline, high-risk coverage rose from 217 to
503 lines (+286), but that is uncertainty detection, not proof that character
accuracy improved.

The deployed target passed 158 tests. The only remaining test warning is the
existing FastAPI/Starlette `httpx` deprecation warning; it does not affect OCR
inference.

## Engine decisions

| Engine/project | CPU/offline fit | Decision |
|---|---|---|
| Existing Paddle detector + VietOCR | Good; already integrated | Keep as primary |
| Tesseract 5.4 + `vie+eng` `tessdata_best` | Excellent CPU speed, independent architecture | Bundle as required verifier |
| PP-OCRv6 medium | Strong multilingual/diacritic model; very slow on this Windows CPU | Benchmark/reference, not request path |
| MinerU 3.4.4 pipeline | Runs quickly after warm-up on CPU, but the CLI exposes no Vietnamese language option and selected `ch_PP-OCRv6_small` | Rejected for Vietnamese legal OCR |
| Docling | CPU orchestration over Tesseract/RapidOCR; useful for layout/tables | Optional benchmark, not a second raw recognizer |
| ScanIndex | Vietnamese admin workflow, CPU-only | Research only: code/license and ScreenAI redistribution terms are not production-safe |
| ABBYY JFK-OCR | Searchable OCR dataset, not an OCR engine | Dataset/research only |
| OCRFlux / olmOCR and page VLMs | GPU-oriented for practical throughput | Research only on this CPU machine |

## Production configuration

The bundled verifier lives under `tools/tesseract` and includes only the
runtime DLLs, `tesseract.exe`, `vie.traineddata`, `eng.traineddata`, configs,
and upstream license/readme. It runs one TSV pass per page and reuses the page
coordinates already produced by the detector.

oneDNN is disabled by default because Paddle 3.3.1 on this Windows environment
throws `ConvertPirAttribute2RuntimeAttribute` during the first detector
inference. The existing oneDNN fallback remains available when explicitly
enabled for a different compatible host.

## MinerU CPU trial

MinerU 3.4.4 was installed in an isolated environment on drive D. Its first
Hugging Face model download failed on Windows cache symlink creation
(`WinError 1314`); switching `MINERU_MODEL_SOURCE=modelscope` and placing the
ModelScope cache on D resolved that deployment issue.

The one-page pipeline trial then completed. Model download/initialization took
230.9 seconds and the warm analysis stages took only a few seconds, but the CLI
had no `vi` value and loaded `ch_PP-OCRv6_small`. The resulting Vietnamese text
was not acceptable: examples include `Luật` → `Lut`, `số` → `s`, `Điều` →
`Đių`, and `ngày 14` → `ngày 44`. Running the remaining 26 pages would not
change this language-model incompatibility, so MinerU was not integrated.

## Relevant upstream material

- MinerU: <https://github.com/opendatalab/MinerU>
- ScanIndex: <https://github.com/welcomyou/scanindex>
- ABBYY JFK-OCR: <https://github.com/abbyy/JFK-OCR>
- Tesseract: <https://github.com/tesseract-ocr/tesseract>
- Tesseract `tessdata_best`: <https://tesseract-ocr.github.io/tessdoc/Data-Files-in-tessdata_best.html>
- PaddleOCR PP-OCRv6: <https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html>
- Docling OCR: <https://docling-project.github.io/docling/concepts/OCR/>
- OmniDocBench: <https://github.com/opendatalab/OmniDocBench>
