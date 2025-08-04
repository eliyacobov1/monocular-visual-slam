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
pipeline on a short synthetic translation clip. The errors drop dramatically
compared to the previous evaluation without alignment:

```text
ATE_RMSE 2.4113
ATE_MEAN 2.0477
ATE_MEDIAN 2.3120
RPE_RMSE 2.1322
RPE_MEAN 2.0175
RPE_MEDIAN 1.8833
```

For reference, running the same sequence without similarity alignment yielded
ATE and RPE errors above 12 pixels. Once alignment is in place you can benchmark
longer TUM sequences to better assess drift and the effect of loop closures.

## Development

Tests require Python 3.11+ with `numpy`, `opencv‑python‑headless`,
`matplotlib`, `scikit‑learn` and `pytest` installed.  Run them with:

```bash
pytest -q
```

The SLAM test synthesises a short translating clip and verifies the estimated
camera translations.

