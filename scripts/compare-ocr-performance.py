from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def document_sessions(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Return the last complete event sequence for each document filename."""
    sessions: dict[str, list[dict[str, Any]]] = {}
    current_name: str | None = None
    current: list[dict[str, Any]] = []
    orphan_pages: list[dict[str, Any]] = []
    for event in read_events(path):
        kind = event.get("event")
        if kind == "document_started":
            current_name = str(event.get("filename", "unknown"))
            current = [event]
            orphan_pages = []
            continue
        if current_name is None:
            if kind == "page_complete":
                orphan_pages.append(event)
            elif kind == "document_complete" and orphan_pages:
                # Pasted/trimmed logs may start in the middle of a document.
                # A complete event gives us a safe filename for the orphan pages.
                name = str(event.get("filename", "unknown"))
                sessions[name] = [*orphan_pages, event]
                orphan_pages = []
            continue
        current.append(event)
        if kind == "document_complete":
            sessions[current_name] = current
            current_name = None
            current = []
    return sessions


def summarize_events(path: Path, filename: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    pages = [e for e in events if e.get("event") == "page_complete"]
    complete = next((e for e in reversed(events) if e.get("event") == "document_complete"), {})
    recognition = [e.get("metrics", {}).get("recognition", {}) for e in pages]
    stage_keys = (
        "pdf_render_ms", "quality_gate_ms", "line_detection_ms", "line_crop_ms",
        "vietocr_ms", "page_total_ms",
    )
    rec_time_keys = (
        "greedy_batch_ms_allocated", "greedy_model_cpu_ms_allocated",
        "decoder_evidence_ms", "decoder_preprocess_wall_ms", "decoder_encoder_wall_ms",
        "decoder_model_wall_ms", "decoder_model_cpu_ms", "decoder_attention_extract_wall_ms",
        "decoder_torch_postprocess_wall_ms", "decoder_visual_grounding_wall_ms",
        "decoder_visual_grounding_cpu_ms", "decoder_trace_build_wall_ms",
        "cross_segment_analysis_ms", "secondary_verifier_ms",
    )
    workload_keys = (
        "crop_count", "segment_count", "shared_batch_count", "padded_segment_count",
        "padding_source_columns", "greedy_batch_fallback_count",
        "greedy_batch_fallback_segment_count", "decoder_evidence_candidate_count",
        "decoder_evidence_seed_selected_count", "decoder_evidence_context_forced_count",
        "decoder_evidence_selected_count", "decoder_evidence_unchecked_candidate_count",
        "decoder_evidence_trace_count",
        "decoder_evidence_trace_batch_count", "decoder_forward_call_count",
        "decoder_sample_step_count", "decoder_attention_element_count",
        "decoder_trace_input_pixel_count", "decoder_trace_character_count",
        "decoder_evidence_cross_segment_candidate_count",
        "decoder_evidence_cross_segment_trimmed_count",
    )
    result: dict[str, Any] = {
        "path": str(path), "filename": filename,
        "document_ms": float(complete.get("processing_time_ms", 0.0) or 0.0),
        "pages": len(pages), "stages": {}, "recognition": {}, "workload": {}, "windows": [],
    }
    for key in stage_keys:
        result["stages"][key] = sum(float(e.get("metrics", {}).get(key, 0.0) or 0.0) for e in pages)
    for key in rec_time_keys:
        result["recognition"][key] = sum(float(r.get(key, 0.0) or 0.0) for r in recognition)
    for key in workload_keys:
        result["workload"][key] = sum(float(r.get(key, 0.0) or 0.0) for r in recognition)

    seen: set[tuple[str, float]] = set()
    for rec in recognition:
        diag = rec.get("performance_diagnostics") or {}
        work = rec.get("workload_fingerprint") or {}
        if not isinstance(diag, dict) or not diag:
            continue
        key = (str(work.get("fingerprint", "")), float(diag.get("recognizer_wall_ms", 0.0) or 0.0))
        if key in seen:
            continue
        seen.add(key)
        result["windows"].append({"diagnostics": diag, "workload": work})
    return result


def pct(new: float, old: float) -> str:
    if old == 0:
        return "n/a"
    return f"{((new / old) - 1.0) * 100:+.1f}%"


def print_section(name: str, left: dict[str, float], right: dict[str, float]) -> None:
    print(f"\n{name}")
    print(f"{'metric':40s} {'baseline':>14s} {'candidate':>14s} {'delta':>9s}")
    for key in sorted(set(left) | set(right)):
        a = float(left.get(key, 0.0) or 0.0)
        b = float(right.get(key, 0.0) or 0.0)
        print(f"{key:40s} {a:14.2f} {b:14.2f} {pct(b, a):>9s}")


def compare(filename: str, a: dict[str, Any], b: dict[str, Any]) -> None:
    print("\n" + "=" * 84)
    print(f"Document : {filename}")
    print(f"Baseline : {a['path']}")
    print(f"Candidate: {b['path']}")
    print(f"Runtime  : {a['document_ms']:.2f} -> {b['document_ms']:.2f} ms ({pct(b['document_ms'], a['document_ms'])})")
    print(f"Pages    : {a['pages']} -> {b['pages']}")
    if a["pages"] != b["pages"]:
        print("WARNING  : page counts differ; stage/workload percentage deltas are not directly comparable")
    print_section("Page-stage totals (ms)", a["stages"], b["stages"])
    print_section("Recognition timing totals (ms)", a["recognition"], b["recognition"])
    print_section("Workload totals", a["workload"], b["workload"])

    print("\nRecognition-window fingerprints")
    for label, value in (("baseline", a), ("candidate", b)):
        print(label + ":")
        if not value["windows"]:
            print("  (performance diagnostics unavailable in this release)")
        for index, window in enumerate(value["windows"], 1):
            d, w = window["diagnostics"], window["workload"]
            start, end = d.get("runtime_start", {}) or {}, d.get("runtime_end", {}) or {}
            print(
                f"  {index}: fp={w.get('fingerprint')} wall={d.get('recognizer_wall_ms')} "
                f"cpu={d.get('recognizer_cpu_ms')} cpu/wall={d.get('cpu_wall_ratio')} "
                f"unaccounted={d.get('unaccounted_wall_ms')} diag={d.get('diagnostics_overhead_ms')} "
                f"freq={start.get('cpu_freq_current_mhz')}->{end.get('cpu_freq_current_mhz')}MHz "
                f"rss={start.get('process_rss_mb')}->{end.get('process_rss_mb')}MB"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OCR runtime/workload logs by document")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--filename", help="Compare only this PDF filename")
    args = parser.parse_args()
    left_sessions = document_sessions(args.baseline)
    right_sessions = document_sessions(args.candidate)
    if args.filename:
        names = [args.filename] if args.filename in left_sessions and args.filename in right_sessions else []
    else:
        names = sorted(set(left_sessions) & set(right_sessions))
    if not names:
        print("No matching complete document sessions found in both logs")
        print("baseline:", sorted(left_sessions))
        print("candidate:", sorted(right_sessions))
        return 1
    for name in names:
        compare(
            name,
            summarize_events(args.baseline, name, left_sessions[name]),
            summarize_events(args.candidate, name, right_sessions[name]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
