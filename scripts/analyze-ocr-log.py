from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


STAGES = (
    "pdf_render_ms",
    "quality_gate_ms",
    "line_detection_ms",
    "line_crop_ms",
    "vietocr_ms",
    "page_total_ms",
)


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Tổng hợp structured OCR log")
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    events = read_events(args.log)
    pages = [event for event in events if event.get("event") == "page_complete"]
    complete = next(
        (event for event in reversed(events) if event.get("event") == "document_complete"),
        None,
    )
    if not pages:
        print("Không tìm thấy event page_complete hợp lệ")
        return 1

    print(f"Số trang có log: {len(pages)}")
    if complete:
        print(f"Document processing_time_ms: {complete.get('processing_time_ms')}")
        print(f"Native/OCR: {complete.get('native_page_count')}/{complete.get('ocr_page_count')}")

    print("\nStage trung bình / median / tổng (ms)")
    for stage in STAGES:
        values = [
            float(event.get("metrics", {}).get(stage, 0.0) or 0.0)
            for event in pages
        ]
        print(
            f"{stage:24s} "
            f"avg={statistics.fmean(values):10.2f} "
            f"median={statistics.median(values):10.2f} "
            f"sum={sum(values):12.2f}"
        )

    recognition = [event.get("metrics", {}).get("recognition", {}) for event in pages]
    counters = (
        "crop_count",
        "segment_count",
        "split_segment_count",
        "wide_crop_count",
        "width_cap_detected_count",
        "width_cap_resolved_count",
        "width_cap_unresolved_count",
        "split_overlap_source_pixels",
        "padded_segment_count",
        "shared_batch_count",
        "seam_overlap_merge_count",
        "tail_segment_count",
        "tail_segment_retry_count",
        "tail_segment_retry_accepted_count",
        "tail_segment_uncertain_count",
        "tail_segment_suppressed_count",
        "hallucination_guard_candidate_count",
        "hallucination_guard_retry_count",
        "hallucination_guard_disagreement_count",
        "hallucination_guard_consensus_count",
        "hallucination_guard_removed_token_count",
        "hallucination_guard_suffix_removed_count",
        "hallucination_guard_prefix_removed_count",
        "hallucination_guard_midline_removed_count",
        "hallucination_guard_numeric_removed_count",
        "hallucination_guard_dominant_ink_retry_count",
        "greedy_batch_fallback_count",
        "greedy_batch_fallback_segment_count",
        "decoder_evidence_candidate_count",
        "decoder_evidence_seed_selected_count",
        "decoder_evidence_context_forced_count",
        "decoder_evidence_selected_count",
        "decoder_evidence_unchecked_candidate_count",
        "decoder_evidence_trace_count",
        "decoder_evidence_trace_batch_count",
        "decoder_evidence_trace_batch_size_max",
        "decoder_evidence_supported_count",
        "decoder_evidence_trace_mismatch_count",
        "decoder_evidence_trace_error_count",
        "decoder_evidence_circuit_breaker_count",
        "decoder_evidence_attention_stall_count",
        "decoder_evidence_visual_exhausted_count",
        "decoder_evidence_near_loop_count",
        "decoder_evidence_midline_span_count",
        "decoder_evidence_midline_trimmed_count",
        "decoder_evidence_line_evidence_count",
        "decoder_evidence_cross_segment_candidate_count",
        "decoder_evidence_cross_segment_trimmed_count",
        "decoder_evidence_cross_segment_rejected_count",
        "decoder_evidence_visual_coverage_exhausted_count",
        "decoder_evidence_attention_reuse_count",
        "secondary_verifier_count",
        "secondary_verifier_primary_extra_count",
        "secondary_verifier_conflict_count",
        "secondary_verifier_ambiguous_count",
        "secondary_verifier_error_count",
        "decoder_evidence_trimmed_count",
        "decoder_evidence_trimmed_char_count",
        "decoder_evidence_suspicious_numeric_count",
        "decoder_evidence_cluster_expansion_count",
        "decoder_evidence_expanded_word_count",
        "beam_retry_count",
        "decoder_loop_detected_count",
        "decoder_partial_loop_detected_count",
        "decoder_loop_trimmed_count",
        "empty_recognition_count",
        "chapter_heading_retry_count",
        "chapter_heading_retry_accepted_count",
        "chapter_heading_retry_unresolved_count",
        "chapter_heading_retry_failed_count",
    )
    print("\nRecognition counters")
    for counter in counters:
        values = [float(value.get(counter, 0) or 0) for value in recognition]
        print(f"{counter:32s} total={sum(values):8.0f} avg/page={statistics.fmean(values):7.2f}")

    print("\nRecognition timing totals (ms)")
    for key in (
        "hallucination_guard_ms",
        "decoder_evidence_ms",
        "decoder_preprocess_wall_ms",
        "decoder_encoder_wall_ms",
        "decoder_model_wall_ms",
        "decoder_attention_extract_wall_ms",
        "decoder_torch_postprocess_wall_ms",
        "decoder_visual_grounding_wall_ms",
        "decoder_trace_build_wall_ms",
        "cross_segment_analysis_ms",
        "greedy_model_cpu_ms_allocated",
        "decoder_model_cpu_ms",
        "decoder_visual_grounding_cpu_ms",
        "secondary_verifier_ms",
        "chapter_heading_retry_ms",
        "tail_segment_retry_ms",
        "beam_retry_ms",
    ):
        values = [float(value.get(key, 0.0) or 0.0) for value in recognition]
        print(f"{key:32s} total={sum(values):10.2f} avg/page={statistics.fmean(values):8.2f}")

    diagnostics = [
        value.get("performance_diagnostics", {})
        for value in recognition
        if isinstance(value.get("performance_diagnostics"), dict)
    ]
    if diagnostics:
        # The same recognition-window diagnostic is repeated on pages from that
        # window. De-duplicate by workload fingerprint + recognizer wall time.
        seen: set[tuple[str, float]] = set()
        unique: list[dict[str, Any]] = []
        for value, rec in zip(diagnostics, recognition):
            workload = rec.get("workload_fingerprint", {})
            key = (str(workload.get("fingerprint", "")), float(value.get("recognizer_wall_ms", 0.0) or 0.0))
            if key not in seen:
                seen.add(key)
                unique.append(value)
        print("\nPerformance diagnostics by recognition window")
        for index, value in enumerate(unique, 1):
            start = value.get("runtime_start", {}) or {}
            end = value.get("runtime_end", {}) or {}
            print(
                f"window={index} wall={float(value.get('recognizer_wall_ms', 0.0)):10.2f} "
                f"cpu={float(value.get('recognizer_cpu_ms', 0.0)):10.2f} "
                f"cpu/wall={float(value.get('cpu_wall_ratio', 0.0)):6.2f} "
                f"unaccounted={float(value.get('unaccounted_wall_ms', 0.0)):9.2f} "
                f"freq={start.get('cpu_freq_current_mhz')}->{end.get('cpu_freq_current_mhz')}MHz "
                f"rss={start.get('process_rss_mb')}->{end.get('process_rss_mb')}MB"
            )

    print("\nPre-recognition crop filter")
    nontext = [event.get("metrics", {}).get("nontext_crop_filter", {}) for event in pages]
    for key in ("input_count", "kept_count", "filtered_count"):
        values = [float(value.get(key, 0) or 0) for value in nontext]
        print(f"{key:32s} total={sum(values):8.0f} avg/page={statistics.fmean(values):7.2f}")
    reason_counts: dict[str, int] = {}
    for value in nontext:
        for reason, count in (value.get("reason_counts", {}) or {}).items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count or 0)
    print("filter reason counts:", json.dumps(reason_counts, ensure_ascii=False, sort_keys=True))

    print("\nLayout counters")
    layouts = [event.get("metrics", {}).get("layout_ordering", {}) for event in pages]
    for key in (
        "baseline_fragment_merge_count",
        "narrow_fragment_dropped_count",
        "column_block_count",
    ):
        values = [float(value.get(key, 0) or 0) for value in layouts]
        print(f"{key:32s} total={sum(values):8.0f} avg/page={statistics.fmean(values):7.2f}")
    for key in (
        "signature_crossing_to_right_count",
        "signature_decorative_polygon_count",
        "signature_prefix_count",
    ):
        totals = [
            sum(float(item or 0) for item in value.get(key, []) or [])
            for value in layouts
        ]
        print(f"{key:32s} total={sum(totals):8.0f} avg/page={statistics.fmean(totals):7.2f}")

    modes: dict[str, int] = {}
    for value in layouts:
        mode = str(value.get("layout_mode", "unknown"))
        modes[mode] = modes.get(mode, 0) + 1
    print("layout_mode counts:", json.dumps(modes, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
