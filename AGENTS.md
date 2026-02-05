# Agent Instructions

## Mission
Deliver a production-grade monocular visual SLAM system with an **accuracy-first
pipeline** on KITTI, backed by a **robust asynchronous ingestion control
plane**, deterministic artifacts, and telemetry suitable for CI regression
analysis.

## Engineering priorities
1. **Accuracy over speed**: prioritize pose estimation, loop closure, and
   trajectory stability improvements ahead of micro-optimizations.
2. **Async control plane mastery**: stage supervisors, circuit breakers,
   deterministic ordering buffers, and structured telemetry with backpressure
   visibility.
3. **KITTI-centric tooling**: preserve KITTI dataset expectations, calibration
   formats, and evaluation metrics.
4. **Determinism**: fixed seeds, explicit configs, and reproducible artifacts.
5. **Roadmap-ready interfaces**: maintain APIs that can expand to multi-camera
   and multi-sensor (IMU/GPS) usage, with C++ hooks for latency-critical
   modules.

## Code quality expectations
- Prefer readable, well-typed Python with clean interfaces and explicit config
  objects.
- Log with structured context (no noisy prints).
- Keep modules separated by responsibility (data, tracking, optimization,
  ingestion, evaluation, telemetry).
- Favor algorithmic improvements and reproducibility over premature
  micro-optimizations.

## Release readiness targets
- CI-ready benchmark harness for KITTI/TUM with ATE/RPE + diagnostics gating.
- Regression thresholds for telemetry latency drift and calibration drift.
- Reproducible demo scripts and persistent run artifacts.
- Failure-isolated ingestion pipeline with backpressure metrics and scale
  summaries.

## Senior-level milestones
- **Control-plane orchestration**: stage-level supervisors, circuit breakers,
  deterministic ordering buffers, and structured latency telemetry wired into
  CI outputs.
- **Graph-optimization modularity**: pluggable solvers, robust-loss selection,
  and deterministic solver snapshots with regression gates on accuracy.
- **Calibration drift automation**: baseline/intrinsics deltas gated in CI with
  artifact retention and visual diagnostics.
- **Relocalization resilience**: multi-injection benchmark pack with
  confidence intervals, recovery latency, and robustness scoring.
- **Telemetry comparison tooling**: cross-run latency drift analysis, CSV
  export APIs, and regression gating for stage-level metrics.
- **Multi-sensor readiness**: extensible interface for IMU/GPS fusion and C++
  hooks for latency-critical tracking.

## Workflow notes
- Update README or docs when behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
