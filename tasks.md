# Tasks

## North-Star Engineering Milestones (Senior-Grade)
1. **Ingestion Control Plane 2.0**
   - Multi-stage supervisors with circuit breakers, deterministic ordering
     buffers, and structured telemetry streaming into CI artifacts.
   - Process-backed decode isolation + adaptive queue sizing policies with
     regression thresholds for latency drift.

2. **Graph Optimization Re-Architecture**
   - Modular backend with pluggable solvers (SE(3)/Sim(3)), robust-loss
     selection, and deterministic solver snapshots for reproducibility.
   - End-to-end regression harness for ATE/RPE and loop-closure stability.

3. **Calibration Drift Gates**
   - Automated intrinsics/baseline delta checks with persisted snapshots and
     per-camera drift dashboards for KITTI/TUM.
   - CI gates on drift thresholds with artifact persistence.

4. **Relocalization Benchmark Pack**
   - Multi-injection tracking-loss evaluations, confidence intervals, and
     recovery-latency regressions wired into CI.

5. **Telemetry Comparison CLI**
   - Cross-run latency drift analysis with CSV/JSON exports, regression gating,
     and visual summaries for stage-level metrics.

6. **Multi-Sensor Expansion Blueprint**
   - IMU/GPS fusion interface with deterministic synchronization, plus C++ hook
     scaffolding for latency-critical tracking/optimization.

## Active Focus (Phase 1)
- Deliver the ingestion control plane overhaul with deterministic ordering,
  circuit-breaker isolation, and process-backed decode execution.
- Expand stress-test coverage for concurrency and race-condition resilience.

## Backlog (High-Impact)
- Loop-closure robustness tuning on KITTI with adversarial relocalization tests.
- Multi-camera calibration ingestion + deterministic alignment tooling.
- Telemetry-driven regression dashboards for release readiness.
