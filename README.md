# Monocular Visual SLAM (Python)

A **monocular, sparse, feature-based SLAM pipeline** built in Python for recovering
camera trajectories from video streams. The system targets **accuracy-first
benchmarking** on standard datasets (KITTI, TUM RGB-D) with loop closure and
pose-graph optimization.

![Demo GIF](docs/media/demo.gif) <!-- [GIF/Image placeholder] -->
![Trajectory Plot](docs/media/trajectory.png) <!-- [GIF/Image placeholder] -->

## Key Features

- **Monocular tracking** with ORB features, robust matching, and RANSAC-based
  essential matrix pose estimation.
- **Adaptive RANSAC thresholds** for motion and loop verification.
- **Keyframe selection** and **sliding-window local bundle adjustment** for
  short-term trajectory refinement.
- **Loop closure detection** via Bag-of-Words (BoW) place recognition with
  geometric verification.
- **Persistent map snapshots + relocalization** to save/load keyframes and
  recover tracking after failures (including on-the-fly map snapshot building
  from keyframes in the SLAM API).
- **Pose-graph optimization** in SE(3), with optional **Sim(3)** loop correction
  to mitigate scale drift in monocular runs.
- **Evaluation harness** for ATE/RPE on KITTI and TUM, plus dataset validation
  helpers.
- **Visualization** through Matplotlib live plots and an optional Next.js
  dashboard.

## System Architecture

1. **Tracking / Front-End**
   - ORB feature detection + matching (`feature_pipeline.py`).
   - Motion estimation with essential matrix + `recoverPose` (`homography.py`).
   - Incremental pose integration into the pose graph.

2. **Local Mapping**
   - Keyframe selection based on motion and match quality (`keyframe_manager.py`).
   - 3D point triangulation and local bundle adjustment (`bundle_adjustment.py`).

3. **Loop Closing / Back-End**
   - BoW place recognition for candidate loops (`loop_closure.py`).
   - Geometric verification (essential matrix / homography) before acceptance.
   - Pose-graph optimization with optional Sim(3) corrections (`pose_graph.py`).

4. **Persistence + Relocalization**
   - Schema-versioned map snapshots and BoW indices (`persistent_map.py`).
   - BoW candidate retrieval plus geometric verification for relocalization.

## Mathematical Foundation

- **Rigid-body motion** on $SE(3)$ using 4×4 homogeneous transforms; rotations are
  parameterized via Rodrigues vectors.
- **Epipolar geometry**: essential matrix $E = [t]_\times R$ with RANSAC for
  outlier rejection.
- **Triangulation** and **reprojection error** minimization in local bundle
  adjustment.
- **Optimization backend**: SciPy `least_squares` (Huber / soft\_l1 losses) for
  bundle adjustment and pose-graph refinement.

## Tech Stack

- **Language**: Python 3.x
- **Core libraries**: OpenCV, NumPy, SciPy
- **Loop closure**: scikit-learn (MiniBatchKMeans + cosine similarity)
- **Visualization**: Matplotlib; optional Next.js 14 dashboard (`frontend/`)

## Installation & Dependencies

```bash
git clone <your-repo-url>
cd monocular-visual-slam
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

If you plan to use BoW loop closure, ensure scikit-learn is installed:

```bash
python -m pip install scikit-learn
```

Optional dashboard:

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_SLAM_WS_URL` if the backend is not running at
`ws://localhost:8000/ws`.

## Dataset Usage

### Quick Demo (Bundled Video)

```bash
python visual_slam_offline_entry_point.py
```

The entry point will download a short demo clip if it is missing.

### KITTI Odometry

Validate the dataset layout:

```bash
python dataset_validation.py --dataset kitti --root /data/kitti --sequence 00
```

Run the evaluation harness (ATE/RPE):

```bash
python evaluation_harness.py --config configs/evaluation/kitti_odometry.json
```

Run the CI benchmark suite (includes baseline regression checks and severity scoring):

```bash
python benchmark_ci_runner.py --config configs/evaluation/ci_benchmark.json
```

The CI runner aggregates regression gate results, computes a severity score
from baseline deltas, and writes a machine-readable summary to the configured
output path (defaults to `reports/ci_benchmark_summary.json`).

You can also use a unified experiment config that captures run metadata,
pipeline settings, and baseline regression checks in one file:

```json
{
  "run": {
    "run_id": "kitti_00_eval",
    "dataset": "kitti",
    "seed": 7,
    "output_dir": "reports/kitti_odometry",
    "use_run_subdir": true
  },
  "pipeline": {
    "feature_type": "orb",
    "motion_ransac_threshold": 0.01,
    "adaptive_ransac": true
  },
  "baseline": {
    "store_path": "reports/baselines.json",
    "key": "kitti_00",
    "thresholds": {
      "ate": { "direction": "lower", "tolerance": 0.05 },
      "rpe": { "direction": "lower", "tolerance": 0.02 },
      "mean_inlier_ratio": { "direction": "higher", "tolerance": 0.03 }
    },
    "telemetry": {
      "thresholds": {
        "telemetry_mean_duration_s": { "direction": "lower", "tolerance": 0.02 },
        "telemetry_stage_feature_detect_p95_duration_s": { "direction": "lower", "tolerance": 0.01 }
      }
    }
  },
  "evaluation": {
    "kitti_root": "/data/kitti",
    "gt_root": "/data/kitti/poses",
    "est_root": "outputs/poses",
    "sequences": ["00"],
    "format": "kitti_odom",
    "rpe_delta": 5
  }
}
```

The evaluation harness also supports estimated trajectories saved as SLAM
artifacts (`.npz` bundles) by setting `est_format` to `"slam_npz"` and pointing
`est_path` at the saved file. For example:

```json
{
  "trajectories": [
    {
      "name": "slam_run",
      "gt_path": "/data/kitti/poses/00.txt",
      "est_path": "runs/slam_trajectory.npz",
      "format": "kitti_odom",
      "est_format": "slam_npz",
      "rpe_delta": 5
    }
  ]
}
```

You can also point at a run directory created by the SLAM API and let the
harness resolve the trajectory artifact automatically via `est_run_dir` and an
optional `est_trajectory` override (defaults to the entry `name`):

```json
{
  "trajectories": [
    {
      "name": "slam_run",
      "gt_path": "/data/kitti/poses/00.txt",
      "est_run_dir": "reports/demo_run",
      "est_trajectory": "slam_trajectory",
      "format": "kitti_odom",
      "rpe_delta": 5
    }
  ]
}
```

When `est_run_dir` is provided and a `diagnostics/frame_diagnostics.json` file
is present, the evaluation harness will attach a per-sequence diagnostics
summary to the report payload and merge the diagnostic metrics (e.g., mean
inlier ratio, method selection ratios) into the sequence metrics.

Baseline thresholds support directional tolerances for diagnostics-heavy
metrics. Use `"direction": "higher"` for metrics that should not decrease (such
as inlier ratios) and `"direction": "lower"` for error metrics (ATE/RPE). You
can also pass a scalar threshold to enforce a max-delta regression check.

Telemetry regression gating is configured under `baseline.telemetry`. The
evaluation harness derives latency metrics (global totals and per-stage
mean/p95/max durations) from the telemetry summary and compares them against
their own baseline entry. If you omit `baseline.telemetry.key`, it defaults to
`<baseline.key>_telemetry`.

You can also use `kitti_dataset.py` directly to iterate frames and parse
calibration.

### TUM RGB-D (RGB Only)

Download and extract a sequence, then convert RGB images to a video:

```bash
ffmpeg -r 30 -i rgb/%04d.png tum_sequence.mp4
```

Run the pipeline and save the estimated trajectory:

```bash
python visual_slam_offline_entry_point.py \
  --video tum_sequence.mp4 \
  --intrinsics_file tum_freiburg1_intrinsics.txt \
  --save_poses estimated.txt
```

Evaluate ATE/RPE against `groundtruth.txt`:

```bash
python evaluate_trajectory.py \
  --gt groundtruth.txt \
  --est estimated.txt \
  --rpe_delta 5 \
  --cols 1,2 \
  --est_cols 0,1 \
  --report metrics.txt
```

## Performance & Evaluation

The repository provides both a direct evaluator (`evaluate_trajectory.py`) and a
configuration-driven harness (`evaluation_harness.py`) that emits JSON/CSV
artifacts for reproducible benchmarks.

For CI-style regression gating, use `benchmark_regression_gate.py` with a gate
config that references evaluation configs and baseline thresholds:

```json
{
  "runs": [
    {
      "name": "kitti_00_smoke",
      "config_path": "kitti_odometry.json",
      "require_baseline": true
    },
    {
      "name": "tum_freiburg1_smoke",
      "config_path": "tum_freiburg1.json",
      "require_baseline": true
    }
  ],
  "output_path": "reports/regression_gate_summary.json",
  "max_workers": 1,
  "fail_fast": true
}
```

```bash
python benchmark_regression_gate.py --config configs/evaluation/regression_gate.json
```

**Results table (fill with your runs):**

| Dataset | Sequence | ATE (m) | RPE (m) | Notes |
| --- | --- | --- | --- | --- |
| KITTI | 00 | TBD | TBD | [placeholder] |
| TUM | freiburg1_xyz | TBD | TBD | [placeholder] |

## License & Acknowledgments

- **License**: Not specified yet (add a `LICENSE` file to formalize usage).
- **Acknowledgments**: KITTI and TUM RGB-D datasets; OpenCV, NumPy, SciPy, and
  scikit-learn.

## SLAM API + Persistence Layer

For programmatic runs, use the high-level API and persistence layer to save
trajectories and metrics with structured metadata:

```python
from pathlib import Path
import numpy as np

from feature_pipeline import FeaturePipelineConfig
from robust_pose_estimator import RobustPoseEstimatorConfig
from slam_api import SLAMSystem, SLAMSystemConfig

config = SLAMSystemConfig(
    run_id="demo_run",
    output_dir=Path("reports"),
    config_path=Path("configs/evaluation/kitti_odometry.json"),
    config_hash="manual",
    intrinsics=np.eye(3),
    feature_config=FeaturePipelineConfig(),
    pose_config=RobustPoseEstimatorConfig(),
)

slam = SLAMSystem(config)
# feed frames + timestamps from your dataset loader
# result = slam.run_sequence(frames, timestamps)
```

The API writes trajectories, metrics, and per-frame diagnostics (including
match/inlier counts, inlier ratio, parallax, model score, and status/failure
metadata) into `reports/<run_id>/` by default, while keeping run metadata
reproducible and inspectable. It also emits a telemetry JSON file under
`reports/<run_id>/telemetry/` with per-stage timing for feature detection,
matching, and pose estimation.

### CLI: Run KITTI sequences with the SLAM API

Use `slam_runner.py` to run the high-level SLAM API against a KITTI sequence
and persist trajectories, metrics, and diagnostics:

```bash
python slam_runner.py \
  --root /path/to/kitti \
  --sequence 00 \
  --camera image_2 \
  --config configs/slam_pipeline.json \
  --output_dir reports \
  --run_id kitti_00_slam \
  --use_run_subdir
```

Pipeline configuration file schema:

```json
{
  "feature_config": {
    "nfeatures": 1500,
    "scale_factor": 1.2
  },
  "pose_config": {
    "min_matches": 40,
    "min_inlier_ratio": 0.3
  }
}
```
