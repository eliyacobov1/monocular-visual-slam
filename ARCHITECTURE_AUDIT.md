# Architecture Audit & Hardest-Path Blueprint (Phase 1 + 2)

## Phase 1: Deep Gap Audit

### Why a Senior Algorithm Engineer would reject the current repo
1. **Fragmented control-plane supervision**
   - The ingestion pipeline relied on a single supervisor loop, limiting
     stage-level telemetry, deterministic backpressure analysis, and
     multi-stage scaling visibility.
2. **No structured control-plane event log**
   - Telemetry was reduced to counters and samples without a deterministic,
     queryable event log for CI regression diagnostics.
3. **Coarse queue scaling signals**
   - Queue scaling decisions were not stage-specific, which obscured
     entry/output pressure and made post-mortems on latency regressions
     ambiguous.
4. **Limited backpressure observability**
   - There was no explicit backpressure event stream to trace capacity
     saturation, which blocks CI gating for ingestion reliability.
5. **Missing stress-oriented supervision tests**
   - No stress tests asserted deterministic ordering, race-condition
     containment, or telemetry sanity under high concurrency.

### Heavy-Lift Subsystem Selection
**Subsystem:** Control-plane orchestration and telemetry instrumentation.

**Justification:** The ingestion control plane is the highest leverage point for
accuracy-first SLAM at scale because it dictates deterministic ordering,
backpressure integrity, and the reliability of downstream tracking and
optimization. Senior-level readiness requires stage-level supervision,
structured telemetry, and regression-ready event logs.

## Phase 2: Architectural & Algorithmic Blueprint

### Big-O Profile
| Component | Previous Complexity | New Complexity | Notes |
| --- | --- | --- | --- |
| Queue tuning decision | O(1) | O(1) | Decisions are per-stage, but the asymptotic order is unchanged. |
| Supervisor loop | O(S) | O(S) | S = number of supervised stages; now explicit instead of implicit. |
| Event log append | O(1) | O(1) | Fixed-capacity ring buffer for deterministic events. |
| Ordering buffer | O(log N) | O(log N) | Heap-backed buffer remains logarithmic, with dedupe tracking. |

### Design Justification
The control-plane overhaul resolves critical bottlenecks:
- **Stage-level autonomy.** Each stage (entry/output) now scales independently,
  making telemetry actionable and tuning reproducible.
- **Deterministic event logging.** The control plane emits structured events for
  queue scaling and backpressure, which directly feed CI regression gates.
- **Concurrency resilience.** Supervisors run at deterministic cadence with
  explicit drain conditions, preventing silent pressure build-up.
- **Telemetry richness.** Per-stage samples, latency traces, and event logs
  enable performance regression analysis without manual instrumentation.

### Target Architecture (Control-Plane Orchestration 3.0)
- **Stage supervisors (Strategy pattern):** Each stage has a dedicated
  supervisor for queue/worker scaling and telemetry emission.
- **Deterministic event log:** Fixed-size ring buffer for control-plane events
  to power CI regression diagnostics.
- **Stage-aware telemetry:** Per-stage queue pressure samples, latency
  samples, and backpressure counters for actionable observability.
- **Ordering buffer hardening:** Dedupe-aware heap buffer that preserves
  deterministic output even under out-of-order completion.

### Expected Outcomes
- Deterministic ingestion ordering with stage-level backpressure visibility.
- Telemetry that supports regression gating for control-plane drift.
- Stress-testable ingestion pipeline that surfaces concurrency failures early.
