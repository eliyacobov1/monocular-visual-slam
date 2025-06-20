# Visual SLAM Pipeline

This repository implements a Python based **visual SLAM** pipeline capable of
estimating camera trajectories from a video sequence. The implementation is
compact yet complete and features:

* **Main entry point** – `visual_slam_offline_entry_point.py` extracts ORB
  features, estimates motion using a RANSAC‑robust homography and visualises
  the trajectory with `VehiclePathLiveAnimator`.
* **Pose‑graph optimisation** – the `pose_graph.py` module maintains a graph of
  camera poses and refines it when loops are detected.
* **Loop closure detection** – `loop_closure.py` implements a simple BoW
  database for recognising previously seen locations.
* Additional helpers for homography estimation and camera intrinsics.

## Repository structure

The most relevant modules are:

* `visual_slam_offline_entry_point.py` – main entry point running the pipeline
* `pose_graph.py` – pose graph data structure and optimisation logic
* `loop_closure.py` – simple BoW database for detecting revisited places
* `homography.py` – feature-based homography estimation utilities
* `cam_intrinsics_estimation.py` – optional camera calibration helpers

## Running the pipeline

The repository ships with a short example clip `sharp_curve.mp4`.  Use the
entry‑point script to process the video and display the recovered trajectory:

```bash
python visual_slam_offline_entry_point.py --video sharp_curve.mp4
```

The script loads frames from the given video, detects ORB keypoints and
estimates planar motion between successive images.  Each transform is appended
to a pose graph and the current vehicle path is shown in a Matplotlib window.
When the BoW database identifies a loop, the graph is optimised and the plot is
updated with the refined trajectory.

Useful flags:

* `--max_frames N` – limit the number of processed frames.
* `--semantic_masking` – mask dynamic regions before feature detection using a
  simple frame‑difference algorithm.

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
    --report metrics.txt
```
The script prints several summary statistics for both metrics. When `--report`
is given, the results are also written to the specified file. Running the
pipeline on the first 100 frames of `freiburg1_xyz` yielded very large errors,
confirming that this simple monocular approach struggles on the dataset:

```text
ATE_RMSE 59335.6007
ATE_MEAN 51310.0229
ATE_MEDIAN 51453.2348
RPE_RMSE 5489.5216
RPE_MEAN 5489.1764
RPE_MEDIAN 5498.2939
```

## Development

Tests require Python 3.11+ with `numpy`, `opencv‑python‑headless`,
`matplotlib`, `scikit‑learn` and `pytest` installed.  Run them with:

```bash
pytest -q
```

The SLAM test synthesises a short translating clip and verifies the estimated
homographies.

