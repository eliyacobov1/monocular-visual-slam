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

## Development

Tests require Python 3.11+ with `numpy`, `opencv‑python‑headless`,
`matplotlib`, `scikit‑learn` and `pytest` installed.  Run them with:

```bash
pytest -q
```

The SLAM test synthesises a short translating clip and verifies the estimated
homographies.

