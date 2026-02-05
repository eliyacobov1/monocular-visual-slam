# Architecture Audit & Hardest-Path Blueprint (Phase 1 + 2)

## Phase 1: Deep Gap Audit

### Why a Senior Algorithm Engineer would reject the current repo
1. **Control-plane coupling**
   - Ingestion responsibilities, telemetry, and failure isolation are tightly
     co-located with decoding logic, limiting observability and extension.
2. **Supervisor signal instability**
   - Queue scaling decisions are based on instantaneous depth ratios, which can
     oscillate under bursty input and lead to noisy scaling behavior.
3. **Limited stage isolation**
   - Circuit breaker and retry logic exist but are not clearly separated from
     stage-level supervision and telemetry responsibilities.
4. **Thin performance evidence**
   - Benchmarks exist, but no dedicated control-plane benchmark that surfaces
     supervisor decisions and memory deltas.
5. **Sparse deterministic control-plane documentation**
   - No single artifact enumerates the bottlenecks, complexity, and intended
     control-plane architecture for new contributors.

### Heavy-Lift Subsystem Selection
**Subsystem:** Asynchronous ingestion control plane (supervisor + telemetry +
backpressure + failure isolation).

**Justification:** This is the most operationally sensitive component for long
KITTI sequences. It governs determinism, failure isolation, and telemetry
quality for the entire pipeline.

## Phase 2: Architectural & Algorithmic Blueprint

### Big-O Profile
| Component | Previous Complexity | New Complexity | Notes |
| --- | --- | --- | --- |
| Queue put/get | O(1) | O(1) | Same algorithmic complexity; smoothed signals remove oscillations. |
| Ordering buffer push/pop | O(log n) | O(log n) | Heap-based ordering remains unchanged. |
| Supervisor decision loop | O(1) per tick | O(1) per tick | EMA smoothing adds constant work per tick. |
| Telemetry sampling | O(1) | O(1) | Bounded deques maintain fixed memory. |

### Design Justification
The ingestion control plane is the highest leverage bottleneck because:
- **Telemetry determines reproducibility.** Without consistent sampling and
  deterministic ordering, regression gates cannot trust output quality.
- **Backpressure stability controls throughput.** EMA smoothing yields stable
  scaling decisions without oscillations, making long KITTI sequences more
  reliable.
- **Failure isolation is a production gate.** Circuit breaker state transitions
  and retry policies must be observable and deterministic.

### Target Architecture (Control Plane 3.0)
- **Supervisor orchestration:** Centralized `ControlPlaneSupervisor` manages
  queue resizing and worker scaling using EMA-smoothed depth ratios.
- **Deterministic ordering buffer:** Heap-backed reorder buffer preserves
  sequence order even under parallel decode workers.
- **Circuit breaker isolation:** Explicit state machine to guard decode failures
  while surfacing telemetry for open/half-open transitions.
- **Telemetry bus:** Structured samples for queue pressure, latency, and scaling
  events, persisted in bounded windows.

### Expected Outcomes
- Stable scaling decisions under bursty input.
- Clear separation of control-plane primitives from decode logic.
- Dedicated benchmark for control-plane throughput and memory deltas.
- Expanded test coverage for supervisor logic, EMA behavior, and circuit
  breaker transitions.
