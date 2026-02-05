# Tasks

## North-Star Engineering Milestones (Senior-Grade)
1. **Async Ingestion Control Plane**
   - Adaptive queues, dynamic worker scaling, and failure isolation with
     structured telemetry summaries.
   - CI gating for throughput/latency regressions plus per-sequence fault
     recovery policies (skip, retry, degrade).

2. **Graph-Optimization Re-Architecture**
   - Modular pose-graph backend with pluggable solvers (SE(3)/Sim(3)), robust
     loss selection per constraint type, and deterministic config snapshots.
   - Solver regression tests + reproducibility artifacts.

3. **Calibration Drift Gates**
   - Automated intrinsics/baseline drift checks with gating thresholds and
     persisted calibration snapshots.
   - CI report attachments for auditability and diagnostics.

4. **Relocalization Benchmark Pack**
   - Multi-injection tracking-loss tests with confidence intervals, latency
     regressions, and recovery success metrics.

5. **Telemetry Comparison CLI**
   - Cross-run latency drift analysis, CSV export, and regression gating for
     stage-level metrics (detect/match/pose).

## Active Focus (Phase 1)
- Ship ingestion control-plane telemetry with adaptive queue tuning, dynamic
  worker scaling, and failure summaries in evaluation artifacts.

## Backlog (High-Impact)
- Multi-camera calibration ingestion + deterministic alignment tooling.
- C++ hook scaffolding for latency-critical tracking or optimization stages.
