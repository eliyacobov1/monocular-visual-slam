# Architecture Audit & Hardest-Path Blueprint (Phase 1 + 2)

## Phase 1: Deep Gap Audit

### Why a Senior Algorithm Engineer would reject the current repo
1. **No deterministic tracking control plane**
   - Async feature extraction is available, but tracking orchestration lacks a
     supervisory layer for frame deadlines, drop policies, and circuit breaker
     recovery.
2. **Missing frame-level backpressure governance**
   - There is no bounded frame buffer with deterministic TTL expiry, which
     makes stall recovery and telemetry gating unreliable under load.
3. **No unified tracking telemetry**
   - Frame wait time, feature queue latency, and drop causes are not aggregated
     into a single drift-ready summary for CI gating.
4. **Tracking resilience is not observable**
   - Drop events, deadline expirations, and breaker trips are not streamed as
     structured events, blocking root-cause diagnostics.
5. **No tracking stress coverage**
   - Concurrency and drop policies for tracking have no dedicated stress
     verification suite.

### Heavy-Lift Subsystem Selection
**Subsystem:** Tracking control plane and deterministic frame supervision.

**Justification:** Tracking is the critical path for pose stability. Without a
 deterministic, supervised tracking control plane, async feature extraction
 cannot be trusted under load. A Senior-level system needs bounded buffers,
 deadline enforcement, and event-driven telemetry for every frame.

## Phase 2: Architectural & Algorithmic Blueprint

### Big-O Profile
| Component | Previous Complexity | New Complexity | Notes |
| --- | --- | --- | --- |
| Pending frame buffer insert | O(1) | O(log N) | Heap-backed TTL tracking adds logarithmic maintenance. |
| Expiry sweep | O(1) | O(K log N) | K = number of expirations; heap pop drives complexity. |
| Feature result merge | O(1) | O(1) | Direct lookup in ordered buffer remains constant time. |
| Event log append | O(1) | O(1) | Fixed-capacity ring buffer. |

### Design Justification
The new tracking control plane resolves critical bottlenecks:
- **Bounded deterministic buffering.** Ordered buffers with TTL ensure that
  stalled frames cannot deadlock the tracking loop.
- **Circuit breaker resiliency.** Consecutive failures trip the breaker and
  protect downstream pose estimation from cascading errors.
- **Telemetry-first supervision.** Aggregated wait-time quantiles and structured
  event logs enable regression gating and drift analysis.
- **Explicit drop governance.** Drop reasons are emitted deterministically,
  enabling actionable SLOs for frame retention.

### Target Architecture (Tracking Control Plane)
- **Pending frame buffer (Heap + OrderedDict):** TTL-backed frame buffer with
  deterministic drop policies.
- **Tracking supervisor (State pattern):** Circuit breaker backed state changes
  (healthy, recovering, tripped).
- **Telemetry aggregator:** Streaming quantiles for end-to-end wait time and
  feature queue latency.
- **Structured event log:** Ring buffer for drop events and orphaned results.

### Expected Outcomes
- Deterministic tracking under backpressure with bounded memory usage.
- CI-ready telemetry drift evaluation for tracking latency regressions.
- Stress-testable tracking control plane with explicit drop behavior.
