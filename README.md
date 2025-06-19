# Computer Vision Offline Demos

Collection of small yet self‑contained algorithms for demonstrating offline
computer vision pipelines:

* **Visual SLAM prototype** (see `visual_slam_offline_entry_point.py`) using
  ORB features and a RANSAC‑robust homography. Results are visualised with
  `VehiclePathLiveAnimator`.
* **Pose‑graph optimiser and loop‑closure utilities** used by the SLAM demo
  (`pose_graph.py`, `loop_closure.py`).

### Development

Tests require Python 3.11+ with `numpy`, `opencv‑python‑headless`,
`matplotlib` and `pytest` installed.  Run them with:

```bash
pytest -q
```

The SLAM test synthesises a short translating clip and verifies the estimated
homographies.

