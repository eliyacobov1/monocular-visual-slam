# Agent Instructions

## Project focus
This repository is a Python-first monocular visual SLAM pipeline. The near-term
focus is **accuracy-first robustness** on KITTI sequences, while progressing
toward production-grade evaluation and ingestion. Longer term, the roadmap
includes multi-camera support and additional sensors (IMU/GPS), with **C++
hooks** for latency-critical components once Python baselines are solid.

## Engineering priorities
1. **Accuracy before speed**: improve pose estimation, loop closure, and
   trajectory stability before micro-optimizations.
2. **KITTI-centric tooling**: align datasets, calibrations, evaluation, and
   benchmarks with KITTI expectations.
3. **Resilient ingestion**: build asynchronous, failure-isolated pipelines for
   long-run sequences.
4. **Determinism**: fixed seeds, explicit configs, and reproducible artifacts.
5. **Roadmap-ready interfaces**: design for multi-camera and multi-sensor
   expansion without breaking core APIs.

## Code quality expectations
- Favor readable, well-typed Python with clear interfaces.
- Log with structured messages; avoid noisy prints.
- Keep modules separated by responsibility (data, tracking, optimization,
  evaluation).
- Avoid premature micro-optimizations; prefer algorithmic improvements.

## Release readiness targets
- CI-ready benchmark harness for KITTI/TUM with ATE/RPE + diagnostics gating.
- Regression thresholds for telemetry latency drift and calibration drift.
- Reproducible demo scripts and persistent run artifacts.
- Failure-isolated ingestion pipeline with backpressure metrics.

## Major technical milestones (Senior-level)
- **Asynchronous ingestion subsystem**: multi-stage, bounded-queue pipeline with
  IO/decode/track separation, backpressure telemetry, and per-sequence isolation.
- **Graph-optimization re-architecture**: modular back-end with pluggable
  solvers, robust losses, and deterministic configs.
- **Calibration drift gates**: automated baseline/intrinsics delta checks wired
  into CI benchmark configs.
- **Relocalization benchmark pack**: multi-injection evaluations with confidence
  intervals and latency regressions.
- **Telemetry comparison CLI**: cross-run latency drift analysis and CSV export.

## Workflow notes
- Update README or docs when behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
