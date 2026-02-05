# Agent Instructions

## Project focus
This repository is a Python-first monocular visual SLAM pipeline. The immediate
focus is **accuracy-first robustness** on KITTI sequences, plus production-grade
**asynchronous ingestion** with telemetry and failure isolation. The longer-term
roadmap targets multi-camera + multi-sensor support (IMU/GPS) and **C++ hooks**
for latency-critical components once Python baselines are validated.

## Engineering priorities
1. **Accuracy before speed**: improve pose estimation, loop closure, and
   trajectory stability before micro-optimizations.
2. **KITTI-centric tooling**: align datasets, calibrations, evaluation, and
   benchmarks with KITTI expectations.
3. **Resilient ingestion**: maintain asynchronous, failure-isolated pipelines
   for long-run sequences with backpressure telemetry.
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
- Failure-isolated ingestion pipeline with backpressure metrics.

## Major technical milestones (Senior-level)
- **Async ingestion hardening**: adaptive queue sizing, dynamic worker scaling,
  structured failure recovery, and throughput/latency telemetry in CI reports.
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
