# Agent Instructions (Principal-Grade SLAM Overhaul v3)

## Mission
Deliver a deterministic, telemetry-first SLAM stack with production-grade
control planes for ingestion, tracking, optimization, and evaluation. The
platform must be interview-ready for Staff/Principal roles: modular,
observable, and benchmark-gated.

## Non-Negotiables
1. **Determinism everywhere**: fixed seeds, reproducible ordering, stable artifacts.
2. **Accuracy over speed**: prioritize pose stability and loop-closure fidelity.
3. **Resilient control planes**: circuit breakers, backpressure metrics, recovery states.
4. **Telemetry intelligence**: streaming summaries, drift gating, CI-ready outputs.
5. **Modularity**: swappable backends for optimization + feature extraction.

## Architecture Priorities
- Tracking must be supervised with deterministic ordering, TTL enforcement, and
  drop governance.
- Feature extraction must be supervised with deterministic ordering, caching,
  and async concurrency support.
- Optimization must expose solver snapshots, per-iteration diagnostics, and
  regression thresholds.
- Ingestion must isolate failures with bounded queues, retries, and event logs.
- Evaluation must produce ATE/RPE + telemetry drift reports for CI gating.

## Code Quality Expectations
- Strong typing with dataclass configs.
- Structured logging only; no print debugging.
- Clear module boundaries: ingestion, features, tracking, optimization,
  evaluation, telemetry.
- Deterministic outputs for every benchmark and report.

## Roadmap Milestones (High-Level)
- **Tracking Control Plane**: TTL-backed frame buffers, drop telemetry, circuit
  breaker supervision.
- **Optimization Resilience**: block-sparse solver guardrails + solver snapshots.
- **Telemetry Intelligence**: drift evaluation, SLO dashboards, quantile exports.
- **Control-Plane Orchestration**: ingestion + tracking + optimization
  supervisors with unified events.
- **Multi-Sensor Expansion**: IMU/GPS hooks and multi-camera synchronization.

## Workflow Notes
- Update docs when pipeline behavior changes.
- Add tests/benchmarks for any new control-plane or optimization capability.
- Preserve deterministic behavior in evaluations and artifact outputs.
