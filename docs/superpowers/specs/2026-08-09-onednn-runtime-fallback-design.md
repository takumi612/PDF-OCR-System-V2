# oneDNN Runtime Fallback Design

Date: 2026-08-09
Status: Approved

## Problem

`PaddleLineDetector` currently falls back from oneDNN only when
`TextDetection` construction raises. With PaddlePaddle 3.3.1, construction
succeeds but the first `predict()` call raises `NotImplementedError` while
converting a PIR `ArrayAttribute<DoubleAttribute>` for the oneDNN executor.
The exception escapes to FastAPI and produces HTTP 500.

## Selected approach

Keep oneDNN enabled when requested. If inference raises the known Paddle
oneDNN/PIR compatibility error, discard the failed processor, construct a new
processor with `enable_mkldnn=False`, and retry the same image exactly once.
Persist the fallback processor so later pages and requests do not retry the
broken backend.

## Component changes

- `PaddleLineDetector` owns the effective detector runtime and fallback state.
- Processor construction is centralized so initial construction and runtime
  fallback use the same arguments and compatibility handling.
- Runtime fallback is eligible only when all conditions hold:
  - Paddle processor was created internally rather than injected by a caller.
  - CPU inference requested oneDNN.
  - `paddle_mkldnn_fallback` is enabled.
  - The exception is the known `ConvertPirAttribute2RuntimeAttribute` oneDNN
    compatibility failure.
- Unrelated inference exceptions propagate unchanged.
- Retry occurs once. Failure of the non-oneDNN retry propagates unchanged.

## Telemetry

`init_metrics` continues to expose the existing fields and records runtime
fallback with:

- `fallback_used=True`
- `fallback_error_type=<original exception type>`
- `mkldnn_effective=False`
- `fallback_phase="predict"`

Constructor-time fallback records `fallback_phase="init"`.

## Tests

Regression tests use the real `PaddleLineDetector` behavior with injected
processor factories:

1. Known oneDNN/PIR failure rebuilds without oneDNN and retries successfully.
2. The fallback processor is reused on the next detection call.
3. An unrelated prediction exception is not swallowed or retried.
4. Existing constructor fallback behavior remains intact.

The existing detector and OCR pipeline tests, full unit suite, Ruff, and a
real Paddle model A/B smoke test must pass before completion.

## Non-goals

- Patching PaddlePaddle binaries.
- Changing HTTP response schemas.
- Changing native PDF routing or VietOCR behavior.
- Automatically downgrading dependencies.
