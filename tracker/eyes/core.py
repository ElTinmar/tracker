from dataclasses import dataclass
import numpy as np
from tracker.core import Tracker, TrackingOverlay, ParamTracking
from typing import Tuple

DTYPE_EYE = np.dtype([
    ('direction', np.float32, (2,)),
    ('direction_global', np.float32, (2,)),
    ('angle', np.float32),
    ('angle_global', np.float32),
    ('centroid_resized', np.float32, (2,)),
    ('centroid_cropped', np.float32, (2,)),
    ('centroid_input', np.float32, (2,)),
    ('centroid_global', np.float32, (2,))
])

@dataclass
class EyesTrackerParamTracking(ParamTracking):
    dyntresh_res: int = 5
    size_lo_mm: float = 0.0
    size_hi_mm: float = 10.0
    thresh_lo: float = 0.2
    thresh_hi: float = 1.0
    crop_dimension_mm: Tuple[float, float] = (2, 2)
    
    @property
    def size_lo_px(self) -> int:
        return self.target_mm2px(self.size_lo_mm)
    
    @property
    def size_hi_px(self) -> int:
        return self.target_mm2px(self.size_hi_mm)
    
    @property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('success', np.bool_),
            ('left_eye', DTYPE_EYE),
            ('right_eye', DTYPE_EYE),
            ('mask',  np.bool_, self.resized_dimension_px[::-1]),
            ('image_processed',  np.float32, self.resized_dimension_px[::-1]),
            ('image_cropped',  self.input_image_dtype, self.crop_dimension_px[::-1]),
            ('pix_per_mm_global', np.float32),
            ('pix_per_mm_input', np.float32),
            ('pix_per_mm_cropped', np.float32),
            ('pix_per_mm_resized', np.float32),
        ])
        return dt

    @property
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
            tracking_param: EyesTrackerParamTracking = EyesTrackerParamTracking(), 
        ) -> None:

        self.tracking_param = tracking_param

class EyesOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: EyesTrackerParamOverlay = EyesTrackerParamOverlay()
        ) -> None:

        self.overlay_param = overlay_param
