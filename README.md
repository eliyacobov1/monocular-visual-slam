# Visual SLAM Pipeline

This repository implements a Python based **visual SLAM** pipeline capable of
estimating camera trajectories from a video sequence. The implementation is
compact yet complete and features:

* **Main entry point** – `visual_slam_offline_entry_point.py` extracts ORB
  features, estimates motion using a RANSAC‑robust essential matrix and visualises
  the trajectory with `VehiclePathLiveAnimator`.
* **Pose‑graph optimisation** – the `pose_graph.py` module maintains a graph of
  camera poses and refines it when loops are detected.
* **Loop closure detection** – `loop_closure.py` implements a simple BoW
  database for recognising previously seen locations.
* Additional helpers for essential‑matrix estimation and camera intrinsics.

## Repository structure

The most relevant modules are:

* `visual_slam_offline_entry_point.py` – main entry point running the pipeline
* `pose_graph.py` – pose graph data structure and optimisation logic
* `loop_closure.py` – simple BoW database for detecting revisited places
* `homography.py` – feature-based geometry estimation utilities
* `cam_intrinsics_estimation.py` – optional camera calibration helpers

## Running the pipeline

The demos download a short example clip on first use.  Use the
entry‑point script to process the video and display the recovered trajectory:

```bash
python visual_slam_offline_entry_point.py
```

The script loads frames from the given video, detects ORB keypoints and
estimates camera motion between successive images using the essential matrix. Each transform is appended
  to a pose graph and the current vehicle path is shown in a Matplotlib window.
When the BoW database identifies a loop, the graph is optimised and the plot is
updated with the refined trajectory.

Useful flags:

* `--max_frames N` – limit the number of processed frames.
* `--semantic_masking` – mask dynamic regions before feature detection using a
  simple frame‑difference algorithm.
* `--intrinsics_file path` – load calibrated intrinsics (`fx fy cx cy`) from a text file.

## Interactive viewer

The `slam_viewer.py` script provides a side-by-side interface that plays the input video while plotting the estimated trajectory. Matched feature points and basic pose diagnostics are drawn on top of each frame.


Launch the viewer with:
```bash
python slam_viewer.py
```

Useful options:

* `--step` – advance frames one-by-one.
* `--show3d` – toggle a 3-D scatter of the path.
* `--intrinsics_file` – supply camera intrinsics (`fx fy cx cy`).

The viewer requires `opencv-python` and `matplotlib`.

## Benchmarking with the TUM RGB‑D dataset

Download one of the TUM RGB‑D sequences from the official website or its
`cvg.cit.tum.de` mirror, e.g.

```bash
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
```

If the server is unreachable due to network restrictions, copy the sequence from
another machine and extract it locally.

The dataset provides RGB images and a `groundtruth.txt` file. Convert the image
sequence into a video before running the pipeline:
The repository already includes `tum_freiburg1_intrinsics.txt` with the official
camera parameters for these sequences. Pass this file to
`visual_slam_offline_entry_point.py` via `--intrinsics_file` so that pose
estimation uses accurate calibration.

```bash
ffmpeg -r 30 -i rgb/%04d.png tum_sequence.mp4
```

Then process the video and save the estimated trajectory:

```bash
python visual_slam_offline_entry_point.py \
    --video tum_sequence.mp4 \
    --log_level INFO \
    --sleep_time 0 \
    --pause_time 0 \
    --intrinsics_file tum_freiburg1_intrinsics.txt \
    --save_poses estimated.txt
```

Only the RGB images are used – depth values are ignored.  When evaluating,
specify the ground truth columns that hold the x and y coordinates:

With both `groundtruth.txt` and `estimated.txt` in place, run
`evaluate_trajectory.py` to compute Absolute Trajectory Error (ATE) and Relative
Pose Error (RPE):

```bash
python evaluate_trajectory.py --gt groundtruth.txt --est estimated.txt \
    --rpe_delta 5 \
    --cols 1,2 \
    --est_cols 0,1 \
    --report metrics.txt
```
The script prints summary statistics for both metrics. When `--report` is
given the results are also written to the specified file. Using the updated
OpenCV‑based pose estimation and similarity aligned evaluation we tested the
pipeline on a short synthetic translation clip.

## Configuration-driven evaluation harness

To make metrics reproducible across runs, the repository now includes a
configuration-driven evaluation harness that emits per-sequence JSON/CSV
artifacts plus an aggregated summary. Use the JSON configs under
`configs/evaluation/` as starting points.

For KITTI odometry:

```bash
python evaluation_harness.py --config configs/evaluation/kitti_odometry.json
```

For TUM RGB-D sequences:

```bash
python evaluation_harness.py --config configs/evaluation/tum_freiburg1.json
```

Each run writes `summary.json`/`summary.csv` and per-sequence reports under the
configured `output_dir`, along with a plain-text metrics file for quick review.
The summary now includes the evaluation config path and a SHA-256 hash of the
config file to make runs reproducible and easier to compare over time.

## KITTI odometry sequences

For KITTI odometry, place the dataset under a root directory (either the
official `sequences/00` style layout or a flat `00` directory). The new
`kitti_dataset.py` helper iterates over frames and parses calibration files:

```python
from pathlib import Path
from kitti_dataset import KittiSequence

sequence = KittiSequence(Path("/data/kitti"), "00", camera="image_2")
for frame in sequence.iter_frames():
    print(frame.index, frame.path, frame.timestamp)
print(sequence.camera_intrinsics())
```

When evaluating KITTI trajectories, use the odometry format flag to extract
translations from the 3x4 pose matrices. JSON and CSV reports are also
available for reproducible benchmarks:

```bash
python evaluate_trajectory.py \
    --gt poses/00.txt \
    --est estimated.txt \
    --format kitti_odom \
    --rpe_delta 5 \
    --json_report kitti_metrics.json \
    --csv_report kitti_metrics.csv
```
