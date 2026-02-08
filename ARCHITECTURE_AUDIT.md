# Architecture Audit & Hardest-Path Blueprint (Phase 1 + 2)

## Phase 1: Deep Gap Audit

### Why a Senior Algorithm Engineer would reject the current repo
1. **No unified control-plane orchestration**
   - Ingestion, feature extraction, tracking, and optimization emit telemetry
     independently without a deterministic, cross-stage control surface.
2. **Event ordering is fragmented**
   - Stage-specific logs exist, but there is no deterministic merge across
     stages to support reproducible diagnostics or CI gating.
3. **Health snapshots are stage-local**
   - Control-plane health is not aggregated into a single, reportable artifact
     suitable for CI regressions or SLO dashboards.
4. **Pipeline reports are incomplete**
   - SLAM runs do not persist control-plane health digests alongside
     trajectories and metrics.
5. **Missing orchestration stress coverage**
   - Thread-safety and ordering stability of cross-stage orchestration are not
     exercised under concurrent load.

### Heavy-Lift Subsystem Selection
**Subsystem:** Unified control-plane orchestration and deterministic reporting.

**Justification:** A Principal-grade stack requires a single, deterministic
control-plane report that merges ingestion/feature/tracking/optimization
telemetry into a CI-ready artifact. Without this, failure triage and regression
analysis remain ad-hoc and non-reproducible.

## Phase 2: Architectural & Algorithmic Blueprint

### Big-O Profile
| Component | Previous Complexity | New Complexity | Notes |
| --- | --- | --- | --- |
| Stage event merge | N/A | O(E log S) | Heap-based k-way merge across S stages and E events. |
| Stage health snapshot | N/A | O(S) | One snapshot per stage. |
| Report digest | N/A | O(E + S) | Deterministic JSON serialization and hashing. |

### Design Justification
The unified control-plane hub resolves core bottlenecks:
- **Deterministic orchestration.** Heap-based merge yields stable ordering across
  stage logs, enabling reproducible diagnostics.
- **Health consolidation.** Stage snapshots share a normalized schema for
  CI gating and SLO dashboards.
- **Persistent report artifacts.** Control-plane reports are stored alongside
  trajectory/metrics outputs for auditability.
- **Thread-safe event buffering.** The event bus is bounded, deterministic, and
  stress-tested under concurrent writes.

### Target Architecture (Control-Plane Hub)
- **Stage adapters:** Provide health snapshots + event streams for each stage.
- **Deterministic event bus:** Bounded buffer with thread-safe writes.
- **K-way merge (Heap):** Merge stage events with stable ordering.
- **Report digest:** SHA-256 digest of normalized JSON for CI regression checks.

### Expected Outcomes
- Deterministic, cross-stage telemetry reports ready for CI gating.
- Unified health snapshots for ingestion/feature/tracking/optimization stages.
- Stress-tested event buffering and ordering stability.
