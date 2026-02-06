# Agent Instructions (Principal-Grade SLAM Overhaul)

## Mission
Deliver a production-grade monocular visual SLAM system that is **accuracy-first**,
telemetry-driven, and determinism-anchored, with modular optimization backends
and a resilient, asynchronous ingestion control plane suitable for regression
and performance gating on KITTI/TUM.

## Engineering Priorities
1. **Accuracy over speed**: prioritize pose stability, loop closure fidelity,
   and robust loss design ahead of micro-optimizations.
2. **Control-plane resilience**: deterministic supervisors, circuit breakers,
   ordering buffers, and explicit backpressure telemetry.
3. **Optimization modularity**: factor-graph abstractions, deterministic solver
   snapshots, and plug-in robust loss strategies.
4. **Telemetry intelligence**: cross-run drift detection, stage-level SLOs,
   JSON/CSV export, and regression gating.
5. **Determinism everywhere**: fixed seeds, explicit configs, and reproducible
   artifacts for every benchmark/CI run.
6. **Roadmap-ready extensibility**: multi-camera and multi-sensor (IMU/GPS)
   hooks with C++ acceleration paths.

## Code Quality Expectations
- Prefer readable, well-typed Python with explicit config objects.
- Use structured logging (no noisy prints).
- Keep modules separated by responsibility: data, tracking, optimization,
  ingestion, evaluation, telemetry.
- Favor algorithmic correctness and reproducibility over premature
  micro-optimizations.
- Surface performance/quality metrics in structured outputs for CI gating.

## Release Readiness Targets
- CI-ready benchmark harness for KITTI/TUM with ATE/RPE + diagnostics gating.
- Deterministic solver snapshots and regression thresholds for telemetry drift.
- Reproducible demo scripts and persistent run artifacts.
- Failure-isolated ingestion pipeline with backpressure metrics and scale
  summaries.

## Senior-Level Milestones
- **Factor-Graph Optimization Engine**: analytic Jacobians for SE(2),
  deterministic numeric Jacobians for SE(3)/Sim(3), and robust loss weighting
  baked into solver kernels.
- **Telemetry Intelligence Layer**: streaming quantiles, drift evaluation,
  and stage-level SLO dashboards.
- **Control-Plane Orchestration**: deterministic supervisors, race-condition
  stress gates, and backpressure event logs.
- **Calibration Drift Automation**: baseline/intrinsics deltas gated in CI with
  retained diagnostics.
- **Relocalization Resilience**: injection benchmark pack with confidence
  intervals, recovery latency, and robustness scoring.
- **Multi-Sensor Expansion**: IMU/GPS fusion interfaces with deterministic
  synchronization and C++ hooks.

## Workflow Notes
- Update documentation when pipeline behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
