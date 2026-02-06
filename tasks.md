# Strategic Roadmap (Executive-Level Milestones)

## Milestone 1: Control-Plane Orchestration 3.0
- Ship stage-level supervisors with deterministic cadence, per-stage telemetry,
  and explicit backpressure event logs consumed by CI gates.
- Add race-condition stress harnesses and deterministic ordering verification
  under peak ingestion load.
- Publish latency SLOs with regression thresholds for entry/decode/output
  stages and automated gate failures.

## Milestone 2: Graph Optimization 2.0
- Expand solver registry to include sparse backends with deterministic solver
  snapshots captured per run.
- Add robust-loss profiles per dataset (KITTI/TUM) with gating thresholds on
  residual distribution drift.
- Provide C++ hooks and solver parity tests for latency-critical optimization.

## Milestone 3: Calibration Drift Automation
- Build intrinsics/baseline drift detectors with retained JSON + plot artifacts
  for every benchmark sequence.
- Gate CI on calibration deltas, with auto-generated remediation reports.

## Milestone 4: Relocalization Resilience Suite
- Create multi-injection benchmark pack with confidence intervals and
  recovery-latency scoring.
- Add deterministic relocalization buffers and failure-isolation metrics.

## Milestone 5: Telemetry Intelligence Layer
- Implement cross-run telemetry diffing (CSV/JSON export) and CI regression
  dashboards.
- Provide stage-level SLO summaries for ingestion, tracking, and optimization,
  with threshold tuning workflows.

## Milestone 6: Multi-Sensor Expansion Blueprint
- Define IMU/GPS fusion interfaces with deterministic synchronization and
  extensible configs.
- Deliver C++ hook scaffolding for latency-critical tracking/optimization
  integration.
