# Agent Instructions (Principal-Grade SLAM Overhaul v4)

## Mission
Deliver a deterministic, telemetry-first SLAM stack with production-grade
control planes for ingestion, feature extraction, tracking, optimization, and
evaluation. The platform must be interview-ready for Staff/Principal roles:
modular, observable, and benchmark-gated.

## Project Readiness
- Status: In progress (Sprint 1 deterministic ordering + data integrity + stability gates complete).

## Non-Negotiables
1. **Determinism everywhere**: fixed seeds, reproducible ordering, stable artifacts.
2. **Accuracy over speed**: prioritize pose stability and loop-closure fidelity.
3. **Resilient control planes**: circuit breakers, backpressure metrics, recovery states.
4. **Telemetry intelligence**: streaming summaries, drift gating, CI-ready outputs.
5. **Modularity**: swappable backends for optimization + feature extraction.

## Architecture Priorities
- Unified control-plane health reporting across ingestion, feature extraction,
  tracking, and optimization.
- Deterministic event ordering with bounded memory and stable digests.
- Stage health snapshots that are suitable for CI gating.
- Evaluations must publish ATE/RPE + drift reports with deterministic formats.
- Every benchmark must emit runtime + memory delta.

## Code Quality Expectations
- Strong typing with dataclass configs.
- Structured logging only; no print debugging.
- Clear module boundaries: ingestion, features, tracking, optimization,
  evaluation, telemetry, orchestration.
- Deterministic outputs for every benchmark and report.

## Roadmap Milestones (High-Level)
- **Control-Plane Orchestration**: multi-stage hub with deterministic event
  ordering, health snapshots, and state dashboards.
- **Optimization Resilience**: solver snapshots, per-iteration diagnostics, and
  regression gates for CI.
- **Telemetry Intelligence**: drift analytics and SLO dashboards with CI export
  bundles.
- **Recovery Hardening**: relocalization and loop-closure validation suites.
- **Multi-Sensor Expansion**: IMU/GPS hooks, multi-camera sync, acceleration
  plan.

## Workflow Notes
- Update docs when pipeline behavior changes.
- Add tests/benchmarks for any new control-plane or optimization capability.
- Preserve deterministic behavior in evaluations and artifact outputs.

## Definition of Done (Senior Interview Ready)
- End-to-end deterministic runs with global seed/config injection and byte-stable artifacts.
- Unified control-plane supervisor with cross-stage state transitions, recovery policies,
  and CI-ready health snapshots.
- Deterministic event ordering and stable digests across ingestion, features, tracking,
  optimization, and evaluation outputs.
- Optimization and loop-closure pipelines emit per-iteration diagnostics, snapshots,
  and regression gates for stability and drift.
- Telemetry intelligence publishes streaming summaries, drift findings, and SLO-ready
  bundles for every run.
- Benchmarks enforce runtime/memory budgets with explicit regression gates and
  deterministic reports.
- Recovery hardening validated via deterministic failure injection and relocalization
  regression suites.
- Documentation and runbooks updated to reflect pipeline control-plane behavior and
  CI expectations.
