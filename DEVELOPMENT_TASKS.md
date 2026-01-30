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
- **Loop-closure verification + robust constraints** (commit `dbbd4ee`): added
  geometric checks (PnP/essential-matrix verification) and robust loop edges to
  improve global trajectory consistency.
- **Scale-drift correction via Sim(3) pose-graph mode**: added a Sim(3)-aware
  pose graph, loop-closure scale estimation, and configuration toggles to
  correct monocular scale drift.
- **Pluggable feature pipeline + adaptive RANSAC** (commit `8b1313a`): added a
  feature/motion interface with ORB defaults, match statistics, and adaptive
  RANSAC thresholds.

## Next task decision (Accuracy + KITTI)
**Decision**: deliver a **dataset integrity + experiment registry layer** that
validates KITTI/TUM inputs, standardizes pipeline configs, and captures
reproducible run artifacts to enable reliable accuracy iteration.

**Rationale**
- The front-end feature pipeline is now modular and adaptive, so the next
  biggest risk to accuracy is inconsistent datasets, mis-specified calibration,
  and missing run provenance.
- Production-grade evaluation requires deterministic inputs and traceable
  outputs; a validation + registry layer unlocks reliable regression tracking
  for KITTI sequences and future multi-sensor extensions.
- Standardized experiment configs keep the codebase ready for multi-camera and
  sensor fusion without rewriting run management later.

**Deliverables**
- A dataset validation CLI that checks KITTI/TUM layout, calibration files,
  timestamp alignment, and frame integrity before pipeline runs.
- A unified experiment config schema (pipeline + evaluation) that seeds
  randomness, captures feature/motion settings, and stores resolved parameters.
- Run artifact registry that writes structured logs, metrics, and config hashes
  to a per-run directory with a unique run ID.
- A lightweight regression baseline store (JSON/CSV) to compare new runs
  against known-good KITTI/TUM results.

**Follow-on task**
- **Scale handling**: explore scene constraints, motion priors, or learned scale
  hints to further reduce monocular drift on long KITTI sequences.

## Production-Grade Tasks (Priority)

### 1) Benchmark Regression Gate (CI + Baselines)
- Add deterministic evaluation configs for short KITTI/TUM sequences.
- Store baseline metrics (ATE/RPE + config hash) and enforce regression
  thresholds in CI.
- Publish run artifacts and summaries for traceability.

### 2) Persistent Map + Relocalization
- Define map serialization for keyframes, descriptors, 3D points, and pose-graph
  edges.
- Implement load/save + relocalization (BoW/descriptor matching + geometric
  verification).
- Add failure recovery hooks for tracking loss.

### 3) Multi-Camera Rig Abstraction + Calibration Tooling
- Add a camera rig model (intrinsics/extrinsics + synchronization).
- Extend ingestion to support stereo/multi-camera datasets (KITTI-style).
- Provide calibration validation/visualization utilities.

## Near-term (Accuracy + KITTI)
- **Dataset integrity + run provenance**:
  - Add a dataset validation command for KITTI and TUM with clear error
    reporting and suggested fixes.
  - Capture per-run metadata (config hash, feature pipeline, calibration) in a
    deterministic run artifact directory.
- **Better feature/motion estimation**:
  - Evaluate alternative feature pipelines (e.g., SuperPoint + SuperGlue) behind
    an interface that still supports ORB.
  - Add match-quality diagnostics (inlier ratio, reprojection stats) to improve
    front-end tuning.
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
  time, and add a lightweight regression gate for KITTI/TUM baselines.
- **Dataset integrity checks**: add a validation command to verify dataset
  layout, calibration files, and timestamp alignment before running pipelines.
- **Documentation**: improve dataset setup instructions and provide a quickstart
  for KITTI evaluation.

## Additional opportunities
- **Bundle adjustment**: integrate a local BA module for improved accuracy.
- **Keyframe management**: add smarter keyframe selection policies.
- **Map representation**: persist sparse map points and metadata for reuse.
- **Failure diagnostics**: debug visualizations for matches, epipolar errors,
  and loop closure verification.

## Decision inputs
- `git log --oneline -n 20` to confirm recent completion of loop-closure
  verification work (commit `dbbd4ee`).
