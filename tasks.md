# Tasks

## Strategic Roadmap (Senior-Grade)
1. **Ingestion Control Plane 3.0**
   - Stage-level supervisors with EMA-smoothed backpressure modeling, circuit
     breakers, deterministic ordering buffers, and structured telemetry streams.
   - Process-backed isolation with deterministic ordering guarantees and
     regression gates on latency drift.

2. **Graph Optimization Modularization**
   - Pluggable SE(3)/Sim(3) solvers, robust-loss selection, and deterministic
     solver snapshots with CI gates for accuracy regressions.
   - Multi-solver A/B harness for KITTI/TUM with ATE/RPE + loop closure
     stability diagnostics.

3. **Calibration Drift Automation**
   - Automated intrinsics/baseline delta checks with retained artifact
     snapshots and per-sequence drift dashboards.
   - CI thresholds with replayable drift diagnostics.

4. **Relocalization Resilience Suite**
   - Multi-injection benchmark pack with confidence intervals, recovery latency,
     and robustness scoring.
   - Deterministic re-entry buffers and failure isolation metrics.

5. **Telemetry Intelligence Layer**
   - Cross-run latency drift comparison with CSV/JSON export APIs and regression
     gating for stage-level metrics.
   - Stage-level SLO summaries for ingestion, tracking, and optimization.

6. **Multi-Sensor Expansion Blueprint**
   - IMU/GPS fusion interfaces, deterministic synchronization, and C++ hook
     scaffolding for latency-critical tracking/optimization.

## Active Execution (Phase 1)
- Complete control-plane refactor with supervisor orchestration, EMA-smoothed
  scaling, and deterministic ordering guarantees.
- Expand concurrency stress testing and race-condition diagnostics.
- Ship benchmark scripts with latency + memory deltas to establish baselines.

## Next Milestones
- Loop-closure robustness tuning on KITTI with adversarial relocalization tests.
- Telemetry-driven regression dashboards for release readiness.
- Multi-camera calibration ingestion with deterministic alignment tooling.
