import numpy as np
import cv2
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