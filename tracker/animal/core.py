from typing import Protocol, Tuple
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay
import numpy as np
from numpy.typing import NDArray
from tracker.core import ParamTracking
import cv2
from .assignment import NoAssignment

@dataclass
class AnimalTrackerParamTracking(ParamTracking):        
    min_size_mm: float = 0
    max_size_mm: float = 100.0
    min_length_mm: float = 0.0
    max_length_mm: float = 0.0
    min_width_mm: float = 0.0
    max_width_mm: float = 0.0
    downsample_factor: float = 0.25
    num_animals: int = 1
    intensity: float = 0.2
    
    @property
    def downsampled_shape(self) -> Tuple[int, int]:
        return (
            int(2*((self.downsample_factor * self.input_image_shape[0])//2)),
            int(2*((self.downsample_factor * self.input_image_shape[1])//2))
        ) 

    @property
    def min_size_px(self) -> int:
        return self.target_mm2px(self.min_size_mm)
    
    @property
    def max_size_px(self) -> int:
        return self.target_mm2px(self.max_size_mm) 
        
    @property
    def min_length_px(self) -> int:
        return self.target_mm2px(self.min_length_mm)
    
    @property
    def max_length_px(self) -> int:
        return self.target_mm2px(self.max_length_mm)

    @property
    def min_width_px(self) -> int:
        return self.target_mm2px(self.min_width_mm)
    
    @property
    def max_width_px(self) -> int:
        return self.target_mm2px(self.max_width_mm)

    @property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('success', np.bool_),
            ('num_animals', int),
            ('centroids_resized', np.float32, (self.num_animals, 2)),
            ('centroids_cropped', np.float32, (self.num_animals, 2)),
            ('centroids_input', np.float32, (self.num_animals, 2)),
            ('centroids_global', np.float32, (self.num_animals, 2)),
            ('downsample_ratio', np.float32),
            ('mask', np.bool_, self.resized_dimension_px[::-1]),
            ('image_processed', np.float32, self.resized_dimension_px[::-1]),
            ('image_downsampled', np.float32, self.downsampled_shape),
            ('pix_per_mm_global', np.float32),
            ('pix_per_mm_input', np.float32),
            ('pix_per_mm_cropped', np.float32),
            ('pix_per_mm_resized', np.float32),
            ('pix_per_mm_downsampled', np.float32)
        ])
        return dt
    
    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)

        
@dataclass
class AnimalTrackerParamOverlay:
    radius_mm: float = 0.1
    centroid_color_BGR: tuple = (128, 255, 128)
    centroid_thickness: int = -1
    id_str_color_BGR: tuple = (255, 255, 255)
    id_str_height_mm = 1.5
    label_offset: int = 10
    alpha: float = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    font_line_type = cv2.LINE_AA

    
class Assignment(Protocol):

    def update(self, centroids: NDArray) -> Tuple[NDArray, NDArray]:
        ...

class AnimalTracker(Tracker):

    def __init__(
            self, 
            assignment: Assignment = NoAssignment(),
            tracking_param: AnimalTrackerParamTracking = AnimalTrackerParamTracking(), 
        ):

        self.tracking_param = tracking_param
        self.assignment = assignment

class AnimalOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: AnimalTrackerParamOverlay = AnimalTrackerParamOverlay()
        ):

        self.overlay_param = overlay_param
