from typing import Optional, Protocol, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay
import numpy as np
from tracker.core import ParamTracking

class Assignment(Protocol):

    def update(self):
        ...
    
    def get_ID(self):
        ...

@dataclass
class AnimalTrackerParamTracking(ParamTracking):        
    source_image_shape: Tuple[int, int] # height, width
    min_animal_size_mm: float = 10.0
    max_animal_size_mm: float = 100.0
    min_animal_length_mm: float = 2.0
    max_animal_length_mm: float = 6.0
    min_animal_width_mm: float = 1.0
    max_animal_width_mm: float = 3.0
    downsample_fullres: float = 0.25
    num_animals: int = 1

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
    def min_animal_size_px(self):
        return self.mm2px(self.min_animal_size_mm)
    
    @property
    def max_animal_size_px(self):
        return self.mm2px(self.max_animal_size_mm) 
        
    @property
    def min_animal_length_px(self):
        return self.mm2px(self.min_animal_length_mm)
    
    @property
    def max_animal_length_px(self):
        return self.mm2px(self.max_animal_length_mm)

    @property
    def min_animal_width_px(self):
        return self.mm2px(self.min_animal_width_mm)
    
    @property
    def max_animal_width_px(self):
        return self.mm2px(self.max_animal_width_mm)
    
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('empty', bool),
            ('num_animals', int),
            ('identities', int, (self.num_animals,)),
            ('indices', int, (self.num_animals,)),
            ('centroids', np.float32, (self.num_animals, 2)),
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
