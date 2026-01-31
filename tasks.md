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
