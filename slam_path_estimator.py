"""Minimal live vehicle path visualiser for offline SLAM demos."""

from __future__ import annotations

import logging
import threading
import time

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class VehiclePathLiveAnimator:
    """Maintain and plot the estimated vehicle trajectory."""

    def __init__(self) -> None:
        self.poses: list[np.ndarray] = [np.eye(3)]
        self.positions: list[np.ndarray] = [np.array([0.0, 0.0])]
        self.new_data_available = False
        self.running = True
        self.lock = threading.Lock()

        # Setup plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.line, = self.ax.plot([], [], 'b-o')
        self.start_scatter = None
        self.end_scatter = None

        # self.ax.set_xlabel('X position')
        # self.ax.set_ylabel('Y position')
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

    def _update_plot_loop(self):
        while self.running:
            if self.new_data_available:
                with self.lock:
                    path = np.array(self.positions)
                    self.line.set_data(path[:, 0], path[:, 1])

                    margin = 1
                    self.ax.set_xlim(np.min(path[:, 0]) - margin, np.max(path[:, 0]) + margin)
                    self.ax.set_ylim(np.min(path[:, 1]) - margin, np.max(path[:, 1]) + margin)

                    if self.start_scatter is None:
                        self.start_scatter = self.ax.scatter(path[0, 0], path[0, 1], color='green', label='Start')
                    if self.end_scatter is not None:
                        self.end_scatter.remove()
                    self.end_scatter = self.ax.scatter(path[-1, 0], path[-1, 1], color='red', label='End')

                    self.new_data_available = False

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

            time.sleep(0.05)  # Sleep a bit to avoid busy-waiting

    def stop(self):
        """Terminate the background plot thread and display the result."""

        self.running = False
        self.update_thread.join()
        plt.ioff()
        plt.show()
