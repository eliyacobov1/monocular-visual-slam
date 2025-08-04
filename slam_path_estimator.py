"""Minimal live vehicle path visualiser for offline SLAM demos."""

from __future__ import annotations

import logging
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np

logger = logging.getLogger(__name__)

class VehiclePathLiveAnimator:
    """Maintain and plot the estimated vehicle trajectory."""

    def __init__(self, ax: Axes | None = None) -> None:
        """Create a live trajectory animator.

        Parameters
        ----------
        ax:
            Optional Matplotlib axis to draw on.  When ``None`` a new figure
            and axis are created automatically.  Supplying an existing axis
            allows the trajectory plot to live alongside other visualisations
            such as the input video stream.
        """

        self.poses: list[np.ndarray] = [np.eye(3)]
        self.optimized_poses: list[np.ndarray] | None = None
        self.positions: list[np.ndarray] = [np.array([0.0, 0.0])]
        self.optimized_positions: list[np.ndarray] | None = None
        self.loop_edges: list[tuple[int, int]] = []
        self.loop_lines: list[Line2D] = []
        self.new_data_available = False
        self.running = True
        self.lock = threading.Lock()

        # Setup plot.  If no axis is supplied we create our own figure,
        # otherwise we piggy back on the provided one so the caller can layout
        # multiple visualisations in a single window.
        plt.ion()
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
        else:
            self.ax = ax
            self.fig = ax.figure
        self.line, = self.ax.plot([], [], "b-o", label="Estimate")
        self.opt_line = None
        self.start_scatter = None
        self.end_scatter = None
        self.heading_arrow = None

        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()
        self.ax.legend()

        # Start live updating in a background thread
        self.update_thread = threading.Thread(target=self._update_plot_loop)
        self.update_thread.start()

    def add_transform(self, R: np.ndarray, t: np.ndarray) -> None:
        """Append a relative pose to the path."""
        logger.debug("Adding transform R=%s t=%s", R.tolist(), t.tolist())

        with self.lock:
            pose_delta = np.eye(3)
            pose_delta[:2, :2] = R[:2, :2]
            pose_delta[:2, 2] = t[:2]

            last_pose = self.poses[-1]
            new_pose = last_pose @ pose_delta

            self.poses.append(new_pose)
            self.positions.append(new_pose[:2, 2])

            self.new_data_available = True

    def set_optimized_poses(self, poses: list[np.ndarray]) -> None:
        with self.lock:
            self.optimized_poses = poses
            self.optimized_positions = [p[:2, 2] for p in poses]
            if self.opt_line is None:
                self.opt_line, = self.ax.plot([], [], 'g--', label='Optimized')
            self.new_data_available = True

    def add_loop_edge(self, i: int, j: int) -> None:
        """Record a detected loop for visualisation."""
        with self.lock:
            self.loop_edges.append((i, j))
            self.new_data_available = True

    def _update_plot_loop(self):
        while self.running:
            if self.new_data_available:
                with self.lock:
                    path = np.array(self.positions)
                    self.line.set_data(path[:, 0], path[:, 1])

                    if self.optimized_positions is not None and self.opt_line is not None:
                        opt = np.array(self.optimized_positions)
                        self.opt_line.set_data(opt[:,0], opt[:,1])

                    margin = 1
                    self.ax.set_xlim(np.min(path[:, 0]) - margin, np.max(path[:, 0]) + margin)
                    self.ax.set_ylim(np.min(path[:, 1]) - margin, np.max(path[:, 1]) + margin)

                    if self.start_scatter is None:
                        self.start_scatter = self.ax.scatter(path[0, 0], path[0, 1], color='green', label='Start')
                    if self.end_scatter is not None:
                        self.end_scatter.remove()
                    self.end_scatter = self.ax.scatter(path[-1, 0], path[-1, 1], color='red', label='End')

                    # draw heading arrow for the latest pose
                    last_pose = self.poses[-1]
                    heading = last_pose[:2, 0]
                    pos = last_pose[:2, 2]
                    if self.heading_arrow is not None:
                        self.heading_arrow.remove()
                    arrow_scale = 0.5
                    self.heading_arrow = self.ax.arrow(
                        pos[0], pos[1],
                        heading[0] * arrow_scale,
                        heading[1] * arrow_scale,
                        head_width=0.3,
                        head_length=0.4,
                        fc="orange",
                        ec="orange",
                        label="Heading",
                    )

                    for line in self.loop_lines:
                        line.remove()
                    self.loop_lines.clear()
                    for idx, (i, j) in enumerate(self.loop_edges):
                        p1, p2 = path[i], path[j]
                        line, = self.ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            "m--",
                            label="Loop" if idx == 0 else "_nolegend_",
                        )
                        self.loop_lines.append(line)
                    self.ax.legend()

                    self.new_data_available = False

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            time.sleep(0.05)  # Sleep a bit to avoid busy-waiting

    def stop(self, save_path: str | None = None) -> None:
        """Terminate the background plot thread and optionally save the final plot."""

        self.running = False
        self.update_thread.join()
        if save_path:
            self.fig.savefig(save_path, bbox_inches="tight")
        plt.ioff()
        plt.show()
