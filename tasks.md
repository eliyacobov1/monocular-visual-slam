# Tasks

## North-Star Engineering Milestones (Senior-Grade)
1. **Async Ingestion Hardening + Backpressure Telemetry**
   - Expand the async pipeline with adaptive queue sizing, dynamic worker
     scaling, and drop/latency histogram reporting.
   - Add per-sequence fault isolation with retry policies (skip vs. retry vs.
     degrade) and structured failure summaries in run artifacts.

2. **Graph-Optimization Re-Architecture**
   - Modularize the pose-graph backend with pluggable solvers (SE(3)/Sim(3)) and
     robust loss selection per constraint type.
   - Add deterministic configuration snapshots and solver regression tests.

3. **Calibration Drift Gates**
   - Automate intrinsics/baseline drift checks in CI benchmark configs with
     gating thresholds and report attachments.
   - Persist calibration snapshots alongside trajectories for auditability.

4. **Relocalization Benchmark Pack**
   - Multi-injection tracking-loss tests with confidence intervals and latency
     regressions.
   - Integrate recovery success metrics into the evaluation harness.

5. **Telemetry Comparison CLI**
   - Cross-run latency drift analysis with CSV export and regression gating.
   - Add alerting thresholds for stage-level regressions (detect/match/pose).

## In-Progress Focus (Phase 1)
- Promote the async ingestion pipeline to first-class evaluation tooling,
  including throughput, queue health, and failure summaries in CI reports.

## Archived (Superseded)
- Trivial task lists were removed in favor of milestone-based planning.
