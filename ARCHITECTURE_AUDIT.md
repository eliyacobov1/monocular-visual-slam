# Architecture Audit & Hardest-Path Blueprint (Phase 1 + 2)

## Phase 1: Deep Gap Audit

### Why a Senior Algorithm Engineer would reject the current repo
1. **Monolithic graph optimization**
   - Pose graph and bundle adjustment are tightly bound to SciPy with no
     pluggable solver strategy, making it impossible to validate solver
     robustness or swap to deterministic/custom solvers.
2. **Missing solver telemetry & artifacts**
   - There is no deterministic snapshot artifact of graph inputs, solver
     settings, or residual norms, which blocks regression gates on accuracy.
3. **Lack of robust-loss configurability**
   - Robust loss selection is implicit and inconsistent, preventing systematic
     tuning of outlier handling for loop closure and relocalization drift.
4. **Sparse numerical safeguards**
   - The optimization stack provides minimal guardrails for conditioning or
     failure visibility, limiting production-grade recovery paths.
5. **No stress-oriented solver validation**
   - There is no stress test coverage for repeated optimization, concurrency,
     or regression stability under noisy edges.

### Heavy-Lift Subsystem Selection
**Subsystem:** Graph-optimization modularity and deterministic solver
infrastructure.

**Justification:** Pose graph quality gates the entire SLAM pipeline. Solver
modularity, robust loss control, and deterministic snapshots are the most
algorithmically demanding path that unlocks regression gates and accuracy-first
iteration.

## Phase 2: Architectural & Algorithmic Blueprint

### Big-O Profile
| Component | Previous Complexity | New Complexity | Notes |
| --- | --- | --- | --- |
| Residual evaluation | O(E) | O(E) | Same per-iteration cost, but now shared across solvers. |
| Jacobian assembly | O(E * d^2) | O(E * d^2) | Block-sparse accumulation keeps the same asymptotic order with structured memory. |
| Normal equation solve | O((N*d)^3) | O((N*d)^3) | Dense solve for now, but isolated behind a solver interface for future sparse solvers. |
| Snapshot generation | O(E + N) | O(E + N) | Deterministic serialization cost scales linearly. |

### Design Justification
The optimization subsystem is the highest-leverage bottleneck because:
- **Accuracy hinges on solver quality.** Loop closure and drift correction are
  only as strong as the solver configuration and robust loss strategy.
- **Regression gating requires determinism.** Snapshot artifacts expose solver
  inputs/outputs, enabling CI to track accuracy drift across changes.
- **Future extensibility depends on modularity.** A solver registry and typed
  interfaces make it possible to bring in C++ or GPU solvers without rewriting
  the pipeline.

### Target Architecture (Graph Optimization 2.0)
- **Solver registry (Strategy pattern):** Pluggable solver backends (SciPy,
  Gauss-Newton) selected at runtime with typed configs.
- **Robust loss system:** Huber/Cauchy/Tukey policies with explicit scales,
  consistent across SE(2)/SE(3)/Sim(3) graphs.
- **Block-sparse linearization:** Edge-local Jacobians aggregated into a
  block-sparse normal equation structure for deterministic solver iteration.
- **Deterministic snapshots:** Serialized artifacts that capture solver config,
  loss config, graph topology, and initial states for CI regression gating.

### Expected Outcomes
- Modular solver experimentation without touching graph construction code.
- Deterministic solver artifacts to fuel accuracy regression gates.
- Higher fidelity loop-closure outcomes via robust loss tuning.
- Stress-testable solver behavior under noisy or adversarial edges.
