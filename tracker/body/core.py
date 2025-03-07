import numpy as np
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay, ParamTracking

@dataclass
class BodyTrackerParamTracking(ParamTracking):
    min_size_mm: float = 10.0
    max_size_mm: float = 100.0
    min_length_mm: float = 2.0
    max_length_mm: float = 6.0
    min_width_mm: float = 1.0
    max_width_mm: float = 3.0
    
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
            ('empty', bool),
            ('heading', np.float32, (2,2)),
            ('centroid', np.float32, (2,)),
            ('centroid_original_space', np.float32, (2,)),
            ('origin', np.float32, (2,)),
            ('angle_rad', np.float32),
            ('mask', np.bool_, self.resized_dimension_px[::-1]),
            ('image', np.float32, self.resized_dimension_px[::-1]),
            ('image_fullres', np.float32, self.crop_dimension_px[::-1]),
        ])
        return dt

body_coordinates = np.dtype([
    ('heading', np.float32, (2,2)),
    ('centroid', np.float32, (2,)),
])

@dataclass
class BodyTrackerParamOverlay:
    pix_per_mm: float = 40.0
    heading_len_mm: float = 1
    heading_color_BGR: tuple = (0,128,255)
    lateral_color_BGR: tuple = (128,64,128)
    thickness: int = 1
    arrow_radius_mm: float = 0.1
    alpha: float = 0.5

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px

    @property
    def heading_len_px(self) -> int:
        return self.mm2px(self.heading_len_mm)

    @property
    def arrow_radius_px(self) -> int:
        return self.mm2px(self.arrow_radius_mm)

class BodyTracker(Tracker):

    def __init__(
            self, 
            tracking_param: BodyTrackerParamTracking, 
        ) -> None:

        self.tracking_param = tracking_param

class BodyOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: BodyTrackerParamOverlay
        ) -> None:

        self.overlay_param = overlay_param
