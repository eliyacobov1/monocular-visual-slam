# Agent Instructions (Principal-Grade SLAM Overhaul)

## Mission
Deliver a production-grade monocular visual SLAM platform that is accuracy-first,
telemetry-driven, and determinism-anchored. The system must ship with modular
optimization backends, resilient ingestion control planes, and CI-ready
benchmarks for KITTI/TUM regression gating.

## Engineering Priorities
1. **Determinism everywhere**: fixed seeds, reproducible artifacts, deterministic ordering.
2. **Accuracy over speed**: pose stability, loop-closure fidelity, robust loss design.
3. **Control-plane resilience**: supervisory orchestration, circuit breakers, backpressure telemetry.
4. **Optimization modularity**: block-sparse solvers, solver snapshots, robust loss strategies.
5. **Telemetry intelligence**: drift detection, SLOs, JSON/CSV exports, CI gating.
6. **Roadmap-ready extensibility**: multi-camera and multi-sensor hooks with C++ acceleration paths.

## Code Quality Expectations
- Prefer readable, well-typed Python with explicit configuration objects.
- Use structured logging (no noisy prints).
- Keep modules separated by responsibility: data, tracking, optimization,
  ingestion, evaluation, telemetry.
- Favor algorithmic correctness and reproducibility over premature
  micro-optimizations.
- Surface performance and quality metrics in structured outputs for CI gating.

## Release Readiness Targets
- CI-ready benchmark harness for KITTI/TUM with ATE/RPE + diagnostics gating.
- Deterministic solver snapshots and regression thresholds for telemetry drift.
- Reproducible demo scripts and persistent run artifacts.
- Failure-isolated ingestion pipelines with backpressure metrics.

## Senior-Level Milestones
- **Optimization Control Plane**: supervised solver runs with retry policies,
  loss-scale escalation, and per-iteration telemetry summaries.
- **Block-Sparse Solver Hardening**: preconditioned CG with stability guardrails
  and regression stress suites.
- **Telemetry Intelligence Layer**: streaming quantiles, drift evaluation, and
  stage-level SLO dashboards.
- **Control-Plane Orchestration**: deterministic supervisors, race-condition
  stress gates, and backpressure event logs.
- **Calibration Drift Automation**: baseline/intrinsics deltas gated in CI with
  retained diagnostics.
- **Relocalization Resilience**: injection benchmark packs with confidence
  intervals, recovery latency, and robustness scoring.
- **Multi-Sensor Expansion**: IMU/GPS fusion interfaces with deterministic
  synchronization and C++ hooks.

## Workflow Notes
- Update documentation when pipeline behavior changes.
- Add tests/benchmarks when improving accuracy or infrastructure.
- Preserve deterministic behavior in evaluations.
