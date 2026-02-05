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
- **Async ingestion control plane**: adaptive queue sizing, dynamic worker
  scaling, structured failure recovery, and throughput/latency telemetry in CI
  reports.
- **Graph-optimization re-architecture**: modular backend with pluggable
  solvers, robust losses, and deterministic config snapshots.
- **Calibration drift gates**: automated baseline/intrinsics delta checks wired
  into CI benchmark configs with artifact persistence.
- **Relocalization benchmark pack**: multi-injection evaluations with
  confidence intervals and latency regressions.
- **Telemetry comparison CLI**: cross-run latency drift analysis and CSV export
  with regression gating.

## Workflow notes
- Update README or docs when behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
