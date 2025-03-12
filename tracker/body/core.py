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
    intensity: float = 0.2
    
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
            ('body_axes', np.float32, (2,2)),
            ('body_axes_global', np.float32, (2,2)),
            ('centroid_resized', np.float32, (2,)),
            ('centroid_cropped', np.float32, (2,)),
            ('centroid_input', np.float32, (2,)),
            ('centroid_global', np.float32, (2,)),
            ('angle_rad', np.float32),
            ('angle_rad_global', np.float32),
            ('mask', np.bool_, self.resized_dimension_px[::-1]),
            ('image_processed', np.float32, self.resized_dimension_px[::-1]),
            ('image_cropped', np.float32, self.crop_dimension_px[::-1]),
            ('pix_per_mm_global', np.float32),
            ('pix_per_mm_input', np.float32),
            ('pix_per_mm_cropped', np.float32),
            ('pix_per_mm_resized', np.float32),
        ])
        return dt

@dataclass
class BodyTrackerParamOverlay:
    heading_len_mm: float = 1
    heading_color_BGR: tuple = (0,128,255)
    lateral_color_BGR: tuple = (128,64,128)
    thickness: int = 1
    arrow_radius_mm: float = 0.1
    alpha: float = 0.5

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
