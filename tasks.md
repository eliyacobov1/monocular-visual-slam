# Tasks

## Deep Sprint: Three-Pillar Plan
1. **Persistence Layer**: Add a run-centric data persistence layer for
   trajectories, metrics, and map snapshots with structured metadata.
2. **SLAM API Wrapper**: Introduce a high-level API that orchestrates the
   pipeline, manages configuration, and persists outputs through the data
   layer.
3. **Robust Pose Suite**: Ship a multi-model pose estimation suite (essential +
   homography) with model selection and diagnostics for accuracy gains.

## Current Focus
- Build the persistence layer with reproducible run artifacts and metrics.
- Wire the SLAM API wrapper to run sequences and save trajectories + metrics.
- Integrate the robust pose estimator into the API and diagnostics pipeline.

## Next Steps
- Expand regression gate configs for KITTI/TUM smoke sequences and wire into CI.
- Add evaluation harness hooks that ingest data-layer artifacts for reporting.

## Completed
- ✅ **SLAM API Wrapper**: Added telemetry instrumentation with persisted
  timing artifacts and flush-on-finalize behavior for reproducible runs.
- ✅ **Telemetry Aggregation**: Added per-stage timing summaries in the
  evaluation harness with aggregate and per-sequence telemetry reporting.
- ✅ **Robust Pose Diagnostics Integration**: Wired frame diagnostics summaries
  into the evaluation harness so per-sequence reports include inlier/parallax
  statistics and model-selection metrics from SLAM runs.

## Follow-up Tasks
- Introduce configurable telemetry sampling/decimation for long KITTI
  sequences to reduce disk footprint while preserving trend data.
- Extend telemetry sinks to support JSONL streaming and pluggable backends
  (e.g., Prometheus export) for large-scale experiments.
- Add regression gate thresholds for telemetry latency regressions alongside
  ATE/RPE baselines.
- Build a telemetry-to-report transformer that merges timing summaries with
  per-sequence error diagnostics for richer evaluation artifacts.
- Add a CLI to compare telemetry summaries across runs and export CSV
  dashboards for experiment tracking.
- Add a diagnostics trend report that correlates per-frame inlier ratios with
  ATE/RPE drift to flag degraded pose quality.
- Introduce a histogram-based diagnostics export (JSON + CSV) for per-method
  inlier distributions to support richer regression analysis.
- Extend evaluation configs to optionally enforce diagnostics-based regression
  thresholds alongside ATE/RPE gates.

## Technical Debt Log
- Telemetry aggregation currently loads full event payloads into memory; add a
  streaming summarizer to handle very long runs without memory pressure.
- Diagnostics summarization currently assumes full-frame diagnostics are loaded
  into memory; migrate to a streaming summarizer for very long runs.

## Project Status (Interview Readiness)
**Status**: Partial. CI-ready benchmark automation, persistent map +
relocalization wiring, and multi-camera rig calibration validation are now
available, but multi-camera ingestion/synchronization and end-to-end
relocalization demos remain the primary gaps.

## Minimal Gaps Checklist
- ✅ Robust per-frame failure boundaries with explicit diagnostics metadata.
- ✅ Streaming diagnostics + telemetry summarizers for long KITTI runs.
- ✅ Regression gate thresholds for diagnostics metrics alongside ATE/RPE.
- ✅ Telemetry latency regression thresholds alongside ATE/RPE baselines.
- ✅ CI-ready benchmark harness wiring for performance regression detection.
- ✅ Persistent map + relocalization pipeline wiring (snapshot build + recovery hooks).
- ✅ Multi-camera rig abstraction + calibration validation for stereo/multi-view datasets.
- ⏳ Multi-camera ingestion + synchronization pipeline for stereo/multi-view datasets.
- ⏳ End-to-end relocalization demo sequence with tracking-loss recovery.

## Follow-up Tasks (Post-Implementation)
- Wire the multi-camera rig abstraction into dataset ingestion with explicit
  sync/latency validation for stereo pairs.
- Add calibration regression gates (baseline drift, intrinsics drift) into the
  benchmark runner for KITTI stereo sequences.
- Add an end-to-end relocalization demo script that loads a saved map snapshot,
  forces a tracking loss, and verifies recovery on KITTI sequences.
