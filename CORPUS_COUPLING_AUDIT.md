# Corpus-coupling audit — 1.3.5.6

## Scope

The audit covers runtime Python source from releases 1.3.0 through 1.3.5.2 and the semantic guard introduced in 1.3.5.4. It distinguishes:

- **Corpus-specific coupling:** an exact token, phrase, identifier or replacement learned from the two regression PDFs.
- **Domain-specific behavior:** a configurable rule for Vietnamese administrative documents.
- **Evidence-based behavior:** decisions based on decoder probability, attention movement and image ink support.

## Findings

1. No release contained a runtime blacklist, correction dictionary or exact text-replacement table for the observed OCR artifacts.
2. Release 1.3.5.2 contained one corpus phrase only inside a source-code docstring explaining a test scenario. It did not participate in execution. The phrase has been removed from runtime source in 1.3.5.4.
3. Suffix trimming, loop detection, seam merge, signature ordering and fragment merging use geometry or model evidence rather than lexical identity.
4. The structural-heading retry previously recognized only one default label. The label is now configurable through `GOVERNMENT_OCR_CHAPTER_HEADING_RETRY_LABELS`; regex input is escaped before compilation.
5. Tests may retain real regression examples, but 1.3.5.4 adds lexical-invariance and randomized-token tests proving that identical evidence produces identical decisions for unrelated words.
6. Numerical thresholds can still overfit a small corpus even without lexical hard-coding. For this reason 1.3.5.4 keeps full semantic coverage, logs proposed/rejected spans, and does not automatically mutate numeric spans.
7. 1.3.5.4 removes the structural coupling to segment-local right anchors by mapping attention to line coordinates and validating weak tails against the next segment. The rule is still driven by image/model evidence rather than known corpus words.

## Runtime safeguards

- No word blacklist or language-model rewrite.
- No automatic token insertion or substitution.
- Midline deletion requires supported left and right anchors.
- The removed span must be weaker than both anchors in probability and ink support.
- Attention must stall or recur inside the span and recover afterwards.
- Numeric spans remain warning-only by default.
- Every applied change records span offsets and evidence statistics.

## Re-running the lexical audit

Create a text file containing one observed artifact per line, then run:

```powershell
python scripts/audit-corpus-coupling.py `
  --terms-file .\runtime\observed-artifacts.txt
```

The audit tool has no built-in artifact list; the caller supplies the terms from any current or future evaluation corpus.

8. 1.3.5.5 replaces the fixed attention-bin coverage threshold with scale-invariant continuous overlap and adds diffuse-attention tests across multiple encoder lengths, reducing structural coupling to synthetic attention sharpness.

9. 1.3.5.6 selective evidence và non-text crop filter chỉ dùng probability/geometry/pixel evidence. Split-line context closure dựa trên `original_index`, không đọc lexical content; context-only traces không được local-mutate.
