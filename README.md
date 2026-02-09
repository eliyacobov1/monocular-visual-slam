# Monocular Visual SLAM (Python)

A **monocular, sparse, feature-based SLAM pipeline** built in Python for recovering
camera trajectories from video streams. The system targets **accuracy-first
benchmarking** on standard datasets (KITTI, TUM RGB-D) with loop closure and
pose-graph optimization.

![Demo GIF](docs/media/demo.gif)
![Trajectory Plot](docs/media/trajectory.png)

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
  from keyframes in the SLAM API), with stable snapshot digests for auditing.
- **Pose-graph optimization** in $SE(3)$, with optional **Sim(3)** loop correction
  to mitigate monocular scale drift.
- **Pluggable graph-optimization solvers** with deterministic snapshots and
  robust-loss selection for regression gating.
- **Evaluation harness** for ATE/RPE on KITTI and TUM, plus dataset validation
  helpers.
- **Multi-camera calibration rig** parsing with baseline validation and
  diagnostics for KITTI stereo setups.
- **Visualization** through Matplotlib live plots and an optional Next.js
  dashboard.
- **Streaming frame ingestion** with bounded buffering for large sequences,
  enabling overlap between IO and compute stages.
- **Async ingestion control plane** with EMA-smoothed adaptive queues, dynamic
  worker scaling, deterministic ordering buffers, retry/backoff policies, and
  circuit-breaker isolation for long KITTI runs.

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
   - Stable snapshot digests embedded in map metadata for reproducibility audits.
   - BoW candidate retrieval plus geometric verification for relocalization.

5. **Async Ingestion Control Plane**
   - EMA-smoothed adaptive queues + deterministic reorder buffers with
     telemetry snapshots (`ingestion_pipeline.py`,
     `ingestion_control_plane.py`).
   - Circuit-breaker isolation + retry/backoff to protect long KITTI runs.
   - Optional process-backed decode executor for failure isolation.

## Mathematical Foundation

- **Rigid-body motion** on $SE(3)$ using 4×4 homogeneous transforms; rotations are
  parameterized via Rodrigues vectors (axis-angle).
- **Epipolar geometry**: essential matrix $E = [t]_\times R$ with RANSAC for
  outlier rejection.
- **Triangulation** and **reprojection error** minimization in local bundle
  adjustment.
- **Optimization backend**: pluggable solver registry with SciPy
  `least_squares` and sparse Gauss-Newton solvers, plus configurable robust
  losses for pose-graph refinement and deterministic solver snapshots.

## Tech Stack

- **Language**: Python 3.x
- **Core libraries**: OpenCV, NumPy, SciPy
- **Loop closure**: scikit-learn (MiniBatchKMeans + cosine similarity)
- **Visualization**: Matplotlib; optional Next.js 14 dashboard (`frontend/`)

## Getting Started (3-Step Quickstart)

1. **Install dependencies**
   ```bash
   git clone <your-repo-url>
   cd monocular-visual-slam
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
   python -m pip install -r requirements.txt
   ```
2. **Pick an interface**
   ```bash
   # Local Matplotlib GUI (default)
   python main.py --gui

   # Same as above (defaults to GUI)
   python main.py

   # Web dashboard (starts WebSocket + HTTP)
   python main.py --web
   ```
3. **Open the dashboard**
   - Local GUI opens automatically.
   - Web dashboard: visit `http://localhost:8001` in your browser.

The demo entry point will download a short sample video if one is missing.

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

The lightweight dashboard is provided by `python main.py --web`. You can also
run the optional Next.js UI:

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

Validate multi-camera synchronization (stereo pairs or multi-view rigs):

```bash
python dataset_validation.py \
  --dataset kitti \
  --root /data/kitti \
  --sequence 00 \
  --cameras image_2,image_3 \
  --sync_tolerance_s 0.003
```

Run the evaluation harness (ATE/RPE):

```bash
python evaluation_harness.py --config configs/evaluation/kitti_odometry.json
```

Generate the deterministic Senior Interview Readiness report (bundles control-plane health,
telemetry summaries, and evaluation metrics):

```bash
python readiness_report.py --config configs/evaluation/readiness_report.json
```

Run the CI benchmark suite (includes baseline regression checks and severity scoring):

```bash
python benchmark_ci_runner.py --config configs/evaluation/ci_benchmark.json
```

Benchmark streaming ingestion throughput + memory delta:

```bash
python benchmark_frame_stream.py --frames 500 --queue_capacity 8
```

Benchmark async ingestion throughput + memory delta:

```bash
python benchmark_async_ingestion.py --frames 500 --workers 2 --queue 32
```

Benchmark pose-graph optimization throughput + memory delta:

```bash
python benchmark_graph_optimization.py --nodes 200 --edges 400 --solver scipy
```

Benchmark async ingestion control plane with executor selection:

```bash
python benchmark_ingestion_control_plane.py --frames 500 --executor thread
```

Benchmark control-plane supervisor scaling and memory delta:

```bash
python benchmark_control_plane_supervisor.py
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
