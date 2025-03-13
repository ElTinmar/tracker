from dataclasses import dataclass
import numpy as np
from tracker.core import Tracker, TrackingOverlay, ParamTracking
from functools import cached_property

DTYPE_EYE = np.dtype([
    ('direction', np.single, (2,)),
    ('direction_global', np.single, (2,)),
    ('angle', np.single),
    ('angle_global', np.single),
    ('centroid_resized', np.single, (2,)),
    ('centroid_cropped', np.single, (2,)),
    ('centroid_input', np.single, (2,)),
    ('centroid_global', np.single, (2,))
])

@dataclass
class EyesTrackerParamTracking(ParamTracking):
    dyntresh_res: int = 20
    size_lo_mm: float = 1.0
    size_hi_mm: float = 10.0
    thresh_lo: float = 0.0
    thresh_hi: float = 1.0
    
    @cached_property
    def size_lo_px(self) -> int:
        return self.target_mm2px(self.size_lo_mm)
    
    @cached_property
    def size_hi_px(self) -> int:
        return self.target_mm2px(self.size_hi_mm)
    
    @cached_property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('left_eye', DTYPE_EYE),
            ('right_eye', DTYPE_EYE),
            ('mask',  np.bool_, self.resized_dimension_px[::-1]),
            ('image_processed',  np.float32, self.resized_dimension_px[::-1]),
            ('image_cropped',  np.float32, self.crop_dimension_px[::-1]),
            ('pix_per_mm_global', np.float32),
            ('pix_per_mm_input', np.float32),
            ('pix_per_mm_cropped', np.float32),
            ('pix_per_mm_resized', np.float32),
        ])
        return dt

    @cached_property
    def failed(self):
        return np.zeros((), dtype=self.dtype)
        
@dataclass
class EyesTrackerParamOverlay:
    eye_len_mm: float = 0.25
    color_left_BGR: tuple = (255, 255, 128)
    color_right_BGR: tuple = (128, 255, 255)
    thickness: int = 1
    arrow_radius_mm: float = 0.1
    alpha: float = 0.5
    
class EyesTracker(Tracker):

    def __init__(
            self, 
            tracking_param: EyesTrackerParamTracking, 
        ) -> None:

        self.tracking_param = tracking_param

class EyesOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: EyesTrackerParamOverlay
        ) -> None:

        self.overlay_param = overlay_param
