# Agent Instructions

## Project focus
This repository is a Python-first monocular visual SLAM pipeline that must be
**accuracy-first** on KITTI sequences while shipping a production-grade,
**asynchronous ingestion control plane** with telemetry, failure isolation, and
reproducible artifacts. The long-term roadmap still targets multi-camera and
multi-sensor (IMU/GPS) expansion with **C++ hooks** for latency-critical
components.

## Engineering priorities
1. **Accuracy before speed**: improve pose estimation, loop closure, and
   trajectory stability ahead of micro-optimizations.
2. **Async ingestion control plane**: adaptive queues, dynamic worker scaling,
   structured failure recovery, and explicit telemetry for long-run sequences.
3. **KITTI-centric tooling**: keep datasets, calibrations, evaluation, and
   benchmarks aligned with KITTI expectations.
4. **Determinism**: fixed seeds, explicit configs, and reproducible artifacts.
5. **Roadmap-ready interfaces**: design for multi-camera and multi-sensor
   expansion without breaking core APIs.

## Code quality expectations
- Favor readable, well-typed Python with clear interfaces.
- Log with structured messages; avoid noisy prints.
- Keep modules separated by responsibility (data, tracking, optimization,
  ingestion, evaluation).
- Avoid premature micro-optimizations; prefer algorithmic improvements.

## Release readiness targets
- CI-ready benchmark harness for KITTI/TUM with ATE/RPE + diagnostics gating.
- Regression thresholds for telemetry latency drift and calibration drift.
- Reproducible demo scripts and persistent run artifacts.
- Failure-isolated ingestion pipeline with backpressure metrics and scale
  summaries.

## Major technical milestones (Senior-level)
- **Control-plane orchestration**: stage-level supervisors, circuit breakers,
  deterministic ordering buffers, and structured latency telemetry wired into
  CI outputs.
- **Graph-optimization modularity**: pluggable solvers, robust-loss selection,
  and deterministic solver snapshots with regression gates on accuracy.
- **Calibration drift automation**: baseline/intrinsics deltas gated in CI with
  artifact retention and visual diagnostics.
- **Relocalization resilience**: multi-injection benchmark pack with confidence
  intervals, recovery latency, and robustness scoring.
- **Telemetry comparison tooling**: cross-run latency drift analysis, CSV export
  APIs, and regression gating for stage-level metrics.
- **Multi-sensor readiness**: extensible interface for IMU/GPS fusion and
  C++ hooks for latency-critical tracking.

## Workflow notes
- Update README or docs when behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
