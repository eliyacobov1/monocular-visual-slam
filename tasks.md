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
