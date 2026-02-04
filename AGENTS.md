# Agent Instructions

## Project focus
This repository is a Python-first monocular visual SLAM pipeline. The near-term
focus is **accuracy**, especially on KITTI sequences, before optimizing for
performance. Longer term, the roadmap includes multi-camera support and
additional sensors (e.g., IMU/GPS), with **C++ hooks** for latency-critical
components once Python baselines are solid.

## Priorities
1. **Accuracy first**: improve pose estimation, robustness, loop closure, and
   trajectory quality before micro-optimizations.
2. **KITTI-centric**: prioritize datasets, calibrations, and evaluation tooling
   that align with KITTI (e.g., stereo calibration parsing, pose benchmarks,
   sequence handling).
3. **Multi-sensor roadmap**: design interfaces so multi-camera and additional
   sensors can be integrated later with minimal rework.
4. **Performance after correctness**: optimize Python implementations only
   after accuracy improves; then design C++ bindings for bottlenecks.

## Development workflow
- Keep changes modular and documented; update README or add docs when behavior
  changes.
- Add or update tests/metrics when improving accuracy or evaluation.
- Prefer deterministic, reproducible pipelines (fixed seeds, explicit configs).
- Maintain clear separation between data handling, feature/motion estimation,
  and optimization modules.

## Code quality expectations
- Favor readable, well-typed Python with clear interfaces.
- Log with structured messages; avoid noisy prints.
- Avoid premature micro-optimizations in Python; focus on algorithmic gains.

## Release readiness targets
- A standard evaluation harness for KITTI and TUM.
- Metrics reporting (ATE, RPE) with consistent config handling.
- Reproducible demo scripts for key sequences.
- Regression gate CLI that enforces baseline thresholds for CI readiness.

## Production-Grade Roadmap (Next Milestones)
- **Benchmark regression gating**: CI-based KITTI/TUM evaluation with
  deterministic configs, baseline metrics storage, and pass/fail thresholds for
  ATE/RPE.
- **Persistent maps + relocalization**: save/load map state, reinitialize
  tracking via place recognition, and recover from tracking loss with
  schema-versioned map snapshots and fail-fast validation.
- **Multi-camera rig abstraction**: standardized intrinsics/extrinsics model,
  calibration validation tooling, and synchronized ingestion for stereo and
  multi-camera datasets.

## Planning
See `DEVELOPMENT_TASKS.md` for detailed near-term and long-term tasks.

## Deep Sprint (Three Pillars)
- **Persistence Layer**: run-centric storage for trajectories, metrics, and map
  snapshots with structured metadata.
- **SLAM API Wrapper**: high-level interface that orchestrates the pipeline and
  persists outputs via the data layer.
- **Robust Pose Suite**: multi-model pose estimation (essential + homography)
  with model selection, diagnostics, and accuracy-first defaults.

## Task Completion Updates
- ✅ **SLAM API Wrapper**: Added telemetry instrumentation with persisted timing
  artifacts and flush-on-finalize behavior for reproducible runs.
- ✅ **Telemetry Aggregation**: Added per-stage timing summaries in the
  evaluation harness with aggregate and per-sequence telemetry reporting.
- ✅ **Robust Pose Diagnostics Integration**: Wired frame diagnostics summaries
  into the evaluation harness so per-sequence reports include inlier/parallax
  statistics and model-selection metrics from SLAM runs.

## Next-Gen Follow-up Tasks
- Add configurable telemetry sampling/decimation for long KITTI sequences to
  control artifact sizes while retaining trends.
- Implement pluggable telemetry sinks (JSONL streaming, Prometheus) for
  large-scale experimentation and CI dashboards.
- Add regression thresholds for telemetry latency drift alongside ATE/RPE
  baseline checks.
- Build a telemetry report aggregator that fuses timing summaries with
  per-sequence error diagnostics for richer evaluations.
- Add a CLI to compare telemetry summaries across runs and export CSV
  dashboards for experiment tracking.
- Add diagnostics trend reporting that correlates per-frame inlier ratios with
  ATE/RPE drift to flag degraded pose quality.
- Introduce histogram-based diagnostics exports (JSON + CSV) for per-method
  inlier distributions to support richer regression analysis.
- Extend evaluation configs to optionally enforce diagnostics-based regression
  thresholds alongside ATE/RPE gates.

## Technical Debt Log
- Telemetry aggregation currently loads events into memory; add a streaming
  summarizer to reduce memory usage on long sequences.
- Diagnostics summarization currently loads full frame diagnostics into memory;
  add a streaming summarizer for very long runs.

## Project Status (Interview Readiness)
**Status**: Partial (improving). Relocalization recovery metrics (success rate +
latency) are now regression-gated alongside ATE/RPE and telemetry baselines,
leaving calibration drift validation and long-run ingestion isolation as the
primary readiness gaps.

## Minimal Gaps Checklist
- ✅ Robust per-frame failure boundaries with explicit diagnostics metadata.
- ✅ Streaming diagnostics + telemetry summarizers for long KITTI runs.
- ✅ Regression gate thresholds for diagnostics metrics alongside ATE/RPE.
- ✅ Telemetry latency regression thresholds alongside ATE/RPE baselines.
- ✅ CI-ready benchmark harness wiring for performance regression detection.
- ✅ Persistent map + relocalization pipeline wiring (snapshot build + recovery hooks).
- ✅ Multi-camera rig abstraction + calibration validation for stereo/multi-view datasets.
- ✅ Multi-camera ingestion + synchronization pipeline for stereo/multi-view datasets.
- ✅ End-to-end relocalization demo sequence with tracking-loss recovery.
- ⏳ Calibration drift regression checks (baseline/intrinsics) wired into CI gate configs.
- ⏳ Asynchronous ingestion + failure isolation for long-running multi-sequence runs.

## Follow-up Tasks (Post-Implementation)
- Extend the relocalization demo to optionally load a persisted map snapshot
  and compare recovery performance across runs.
- Add calibration drift regression gates (baseline + intrinsics deltas) into
  KITTI CI benchmark configs.
- Add a relocalization benchmark pack that aggregates success/latency across
  multiple tracking-loss injections per sequence for higher statistical power.
- Introduce confidence-interval reporting for relocalization latency deltas to
  reduce sensitivity to single-run variance.
