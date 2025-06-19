# Computer Vision Offline Demos

Collection of small yet self‑contained algorithms for demonstrating offline
computer vision pipelines:

* **Visual SLAM prototype** (see `visual_slam_offline_entry_point.py`) using
  ORB features and a RANSAC‑robust homography.  Results are visualised with
  `VehiclePathLiveAnimator`.
* **Advanced lane detection** with the YOLOP deep model and IPM warping
  (`advanced_lane_detection.py`).
* **Pure NumPy lane detector** implementing Gaussian blur, Sobel edges and
  Hough transform (`lane_detection.py`).
* **BEV utilities** for ground plane homography and bird’s‑eye view generation
  (`bev.py`).
* **Occupancy grid mapping** from depth images (`free_space.py`).

### Development

Tests require Python 3.11+ with `numpy`, `opencv‑python‑headless`,
`matplotlib` and `pytest` installed.  Run them with:

```bash
pytest -q
```

The SLAM test synthesises a short translating clip and verifies the estimated
homographies.

