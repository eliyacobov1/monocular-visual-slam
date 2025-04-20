import numpy as np
import matplotlib.pyplot as plt

class VehiclePathEstimator:
    def __init__(self):
        self.poses = [np.eye(3)]  # Homogeneous pose matrices (3x3)
        self.positions = [np.array([0.0, 0.0])]  # Starting position in world

    def update(self, R, t):
        """
        Update the global pose using new rotation and translation
        R: 3x3 rotation matrix from frame i to i+1
        t: 3x1 translation vector from frame i to i+1
        """
        # Build transformation matrix for this step (SE2 approximation)
        pose_delta = np.eye(3)
        pose_delta[:2, :2] = R[:2, :2]
        pose_delta[:2, 2] = t[:2]

        # Update current global pose
        last_pose = self.poses[-1]
        new_pose = last_pose @ pose_delta

        self.poses.append(new_pose)

        # Extract X, Y from the new global pose
        position = new_pose[:2, 2]
        self.positions.append(position)

    def get_positions(self):
        return np.array(self.positions)

    def plot_path(self):
        path = self.get_positions()
        plt.figure(figsize=(8, 6))
        plt.plot(path[:, 0], path[:, 1], 'b-o', label='Vehicle Path')
        plt.scatter(path[0, 0], path[0, 1], color='green', label='Start')
        plt.scatter(path[-1, 0], path[-1, 1], color='red', label='End')
        plt.axis('equal')
        plt.title('Estimated Vehicle Path')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.legend()
        plt.grid(True)
        plt.show()
