import numpy as np
import cv2
import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class DrivableAreaBuilder:
    def __init__(self, road_class_id: int = 1, mask_shape: Tuple[int, int] = (480, 640)):
        """
        :param road_class_id: Label index in segmentation mask corresponding to 'road'
        :param mask_shape: Shape of the output mask (H, W)
        """
        self.road_class_id = road_class_id
        self.mask_shape = mask_shape

    def extract_road(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Extract binary road mask from segmentation.
        """
        assert segmentation_mask.shape == self.mask_shape, "Segmentation mask shape mismatch"
        road_mask = (segmentation_mask == self.road_class_id).astype(np.uint8)
        return road_mask

    def apply_obstacle_mask(self, road_mask: np.ndarray, detections: List[Tuple[int, int, int, int]],
                            inflation: int = 5) -> np.ndarray:
        """
        Overlays obstacle detections on road mask and zeroes those pixels.
        
        :param road_mask: Binary mask from extract_road()
        :param detections: List of (x1, y1, x2, y2) tuples
        :param inflation: Pixels to inflate obstacle boxes for safety margin
        """
        mask = road_mask.copy()

        for (x1, y1, x2, y2) in detections:
            # Inflate boxes
            x1i = max(0, x1 - inflation)
            y1i = max(0, y1 - inflation)
            x2i = min(self.mask_shape[1], x2 + inflation)
            y2i = min(self.mask_shape[0], y2 + inflation)
            mask[y1i:y2i, x1i:x2i] = 0

        return mask

    def build_mask(self, segmentation_mask: np.ndarray,
                   detections: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Full pipeline to create binary drivable area mask.
        """
        road_mask = self.extract_road(segmentation_mask)
        final_mask = self.apply_obstacle_mask(road_mask, detections)
        return final_mask

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, rows, cols):
    i, j = pos
    neighbors = []
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < rows and 0 <= nj < cols:
            neighbors.append((ni, nj))
    return neighbors

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(current, rows, cols):
            if grid[neighbor] == 0:
                continue
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

# Example usage:
# 1 = road, 0 = obstacle
grid = np.zeros((100, 200), dtype=int)
grid[40:90, 50:150] = 1  # Simulated road
grid[60:70, 100:110] = 0  # Obstacle

start = (85, 60)
goal = (45, 140)

path = astar(grid, start, goal)

# Visualization
plt.imshow(grid, cmap='gray')
if path:
    path_y, path_x = zip(*path)
    plt.plot(path_x, path_y, color='red')
plt.scatter(start[1], start[0], c='green', marker='o', label='Start')
plt.scatter(goal[1], goal[0], c='blue', marker='x', label='Goal')
plt.legend()
plt.title("A* Path on Drivable Area")
plt.show()