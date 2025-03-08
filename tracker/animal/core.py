from typing import Protocol, Tuple
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay
import numpy as np
from numpy.typing import NDArray
from tracker.core import ParamTracking

@dataclass
class AnimalTrackerParamTracking(ParamTracking):        
    source_image_shape: Tuple[int, int] # height, width
    min_size_mm: float = 10.0
    max_size_mm: float = 100.0
    min_length_mm: float = 2.0
    max_length_mm: float = 6.0
    min_width_mm: float = 1.0
    max_width_mm: float = 3.0
    downsample_fullres: float = 0.25
    num_animals: int = 1
    intensity: float = 0.2

    @property
    def image_shape(self) -> Tuple[int, int]:
        # some video codec require height, width to be divisible by 2
        return (
            int(2*((self.resize * self.source_image_shape[0])//2)),
            int(2*((self.resize * self.source_image_shape[1])//2))
        ) 
    
    @property
    def downsampled_shape(self) -> Tuple[int, int]:
        return (
            int(2*((self.downsample_fullres * self.source_image_shape[0])//2)),
            int(2*((self.downsample_fullres * self.source_image_shape[1])//2))
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

    
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('num_animals', int),
            ('centroids_resized', np.float32, (self.num_animals, 2)),
            ('centroids_cropped', np.float32, (self.num_animals, 2)),
            ('centroids_input', np.float32, (self.num_animals, 2)),
            ('centroids_global', np.float32, (self.num_animals, 2)),
            ('mask', np.bool_, self.image_shape),
            ('image', np.float32, self.image_shape),
            ('image_fullres', np.float32, self.downsampled_shape),
            ('downsample_ratio', np.float32)
        ])
        return dt
        
@dataclass
class AnimalTrackerParamOverlay:
    pix_per_mm: float = 40.0
    radius_mm: float = 0.1
    centroid_color_BGR: tuple = (128, 255, 128)
    centroid_thickness: int = -1
    id_str_color_BGR: tuple = (255, 255, 255)
    label_offset: int = 10
    alpha: float = 0.5

    def mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 

    @property
    def radius_px(self):
        return self.mm2px(self.radius_mm)
    

class Assignment(Protocol):

    def update(self, centroids: NDArray) -> Tuple[NDArray, NDArray]:
        ...


class AnimalTracker(Tracker):

    def __init__(
            self, 
            assignment: Assignment,
            tracking_param: AnimalTrackerParamTracking, 
        ):

        self.tracking_param = tracking_param
        self.assignment = assignment

class AnimalOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: AnimalTrackerParamOverlay
        ):

        self.overlay_param = overlay_param
