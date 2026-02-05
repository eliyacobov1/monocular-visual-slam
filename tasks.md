# Tasks

## North-Star Engineering Milestones (Senior-Grade)
1. **Asynchronous Ingestion + Failure Isolation**
   - Build a multi-stage ingest pipeline (IO → decode → tracking) with bounded
     queues, backpressure metrics, and fault isolation per sequence.
   - Add structured failure reporting with recovery policies (skip, retry,
     fallback) and telemetry for dropped/late frames.

2. **Calibration Drift Regression Gates**
   - Introduce automated calibration drift checks (baseline + intrinsics deltas)
     in CI benchmark configs with pass/fail thresholds.
   - Persist calibration reports alongside trajectory metrics for auditability.

3. **High-Fidelity Optimization Back-End**
   - Introduce a modular graph-optimization layer with pluggable solvers
     (SE(3) + Sim(3)) and robust loss selection per constraint type.
   - Provide a correctness-first optimization suite with deterministic seeding.

4. **Relocalization Benchmark Pack**
   - Expand relocalization evaluation with multiple tracking-loss injections
     per sequence, statistical confidence intervals, and latency regressions.

5. **Telemetry-First Performance Profiling**
   - Provide a dedicated telemetry comparison CLI to detect latency drift across
     runs, export CSV dashboards, and enforce regression thresholds.

## Implementation Roadmap (Near-Term)
- Integrate streaming frame ingestion into the SLAM runner and expose queue
  capacity + telemetry in the evaluation harness.
- Add benchmark scripts to measure throughput and memory deltas for streaming
  ingestion.
- Extend the evaluation harness to surface ingestion drop rates and latency
  distributions.

## Archived (Superseded)
- Trivial task lists were removed in favor of milestone-based planning.
