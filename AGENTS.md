# Agent Instructions (Senior-Grade SLAM)

## Mission
Deliver a production-grade monocular visual SLAM system with **accuracy-first**
performance on KITTI/TUM, deterministic solver artifacts, and a resilient
asynchronous ingestion control plane that is regression-gate ready.

## Engineering priorities
1. **Accuracy over speed**: improve pose estimation, loop closure, and
   trajectory stability ahead of micro-optimizations.
2. **Control-plane resilience**: stage supervisors, circuit breakers,
   deterministic ordering buffers, and structured telemetry with explicit
   backpressure visibility.
3. **Graph-optimization modularity**: pluggable solver interfaces, robust-loss
   selection, and deterministic solver snapshots for regression gating.
4. **Determinism everywhere**: fixed seeds, explicit configs, reproducible
   artifacts for every benchmark run.
5. **Roadmap-ready interfaces**: APIs prepared for multi-camera and
   multi-sensor (IMU/GPS) expansion, with C++ hooks for latency-critical
   modules.

## Code quality expectations
- Prefer readable, well-typed Python with explicit config objects.
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
- **Control-plane orchestration**: deterministic stage supervisors, backpressure
  event logs, and race-condition stress gates.
- **Graph-optimization modularity**: solver registry with deterministic
  snapshots and accuracy regression gates.
- **Calibration drift automation**: baseline/intrinsics deltas gated in CI with
  retained diagnostics.
- **Relocalization resilience**: injection benchmark pack with confidence
  intervals, recovery latency, and robustness scoring.
- **Telemetry intelligence**: cross-run latency drift analysis, CSV/JSON export
  APIs, and regression gating for stage-level metrics.
- **Multi-sensor readiness**: extensible interfaces for IMU/GPS fusion and C++
  hooks for latency-critical tracking.

## Workflow notes
- Update documentation when pipeline behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
