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

## Next-Gen Follow-up Tasks
- Aggregate telemetry into evaluation reports with per-stage timing summaries
  and regression thresholds for performance drift.
- Add configurable telemetry sampling/decimation for long KITTI sequences to
  control artifact sizes while retaining trends.
- Implement pluggable telemetry sinks (JSONL streaming, Prometheus) for
  large-scale experimentation and CI dashboards.

## Technical Debt Log
- Telemetry recorder currently stores events in memory before flushing; add a
  streaming writer to reduce memory usage on long sequences.
