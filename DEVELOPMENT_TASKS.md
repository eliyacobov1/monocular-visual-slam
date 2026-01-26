# Development Tasks

This document tracks the roadmap for improving accuracy, dataset coverage, and
performance. The near-term focus is **accuracy on KITTI** while keeping the
system extensible for multi-camera and additional sensors. Performance and C++
acceleration follow once accuracy targets are met.

## Recently completed
- **KITTI data loader**: add sequence ingestion for KITTI raw/odometry formats,
  including calibration parsing and synchronized frame iteration.
- **Config-driven evaluation harness**: reproducible metrics runs with
  JSON/CSV reporting and config hashing for traceability.
- **Keyframe management + local bundle adjustment**: introduced a keyframe
  selection policy and a sliding-window BA to improve short-term pose stability.

## Next task decision (Accuracy + KITTI)
**Decision**: implement **geometric loop-closure verification + robust loop
constraints** before expanding feature pipelines.

**Rationale**
- The current BoW loop detection does not enforce geometric consistency, which
  can introduce incorrect loop edges and degrade global trajectory accuracy.
- Adding verification (PnP + RANSAC or essential-matrix checks) plus robust
  constraint handling in the pose graph directly targets KITTI evaluation
  robustness without changing the feature pipeline.

**Deliverables**
- Geometric verification stage for loop candidates (PnP + RANSAC or essential
  matrix checks) with clear acceptance thresholds and per-candidate diagnostics.
- Loop-closure constraints added only after verification passes, with a robust
  loss or switchable constraints to downweight residual outliers.
- Evaluation harness updates to record loop-closure impact on KITTI ATE/RPE and
  persist per-run metadata (including verification stats).

**Follow-on task**
- **Scale handling**: add a Sim(3)-aware pose-graph mode or a scale-drift
  correction module that can align sub-trajectories on KITTI sequences.

## Near-term (Accuracy + KITTI)
- **Better feature/motion estimation**:
  - Evaluate alternative feature pipelines (e.g., SuperPoint + SuperGlue) behind
    an interface that still supports ORB.
  - Improve outlier rejection (adaptive RANSAC thresholds, robust loss).
- **Scale handling**: research strategies for monocular scale drift
  compensation (scene constraints, motion priors, or learning-based scale hints).
- **Trajectory evaluation**: standardize ATE/RPE output for KITTI with a clear
  JSON/CSV report and reproducible configs (see `evaluation_harness.py` and
  `configs/evaluation/` for the current harness).
- **Loop closure accuracy**: enhance place recognition (e.g., better BoW vocab,
  TF-IDF weighting, or NetVLAD-like descriptors).

## Mid-term (Multi-camera + Sensor Fusion)
- **Multi-camera ingestion**: support synchronized stereo or surround cameras
  with per-camera intrinsics/extrinsics.
- **Multi-view geometry**: design a camera rig model and update pose estimation
  to use shared motion across camera streams.
- **Optional IMU/GPS fusion**: define a sensor interface and add an estimator
  (e.g., factor graph) that can incorporate IMU/GPS priors when available.
- **Calibration tooling**: add utilities to validate and visualize camera and
  rig calibration for KITTI-style datasets.

## Performance + C++ hooks
- **Profiling baseline**: add a repeatable performance benchmark (frames/sec,
  end-to-end latency, and per-module timing).
- **Vectorize hotspots**: optimize feature extraction/matching and pose graph
  operations in Python before moving to C++.
- **C++ extension plan**: define a stable Python API for C++ acceleration
  targets (feature matching, graph optimization, or bundle adjustment).
- **Interop scaffolding**: pick a binding approach (pybind11/ctypes) and add a
  minimal example module for future scaling.

## Reliability + Production readiness
- **Deterministic runs**: centralized config system with seeded randomness.
- **Robust logging**: structured logs, traceable run IDs, and artifacts per run.
- **CI & tests**: add unit tests for geometry utilities and integration tests
  for short sequences.
- **Regression tracking**: store evaluation metrics to compare changes over
  time.
- **Documentation**: improve dataset setup instructions and provide a quickstart
  for KITTI evaluation.

## Additional opportunities
- **Bundle adjustment**: integrate a local BA module for improved accuracy.
- **Keyframe management**: add smarter keyframe selection policies.
- **Map representation**: persist sparse map points and metadata for reuse.
- **Failure diagnostics**: debug visualizations for matches, epipolar errors,
  and loop closure verification.
