from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from typing import Tuple
from tracker.core import Tracker, TrackingOverlay, ParamTracking

DTYPE_EYE = np.dtype([
    ('direction', np.single, (2,)),
    ('angle', np.single),
    ('centroid', np.single, (2,)),
    ('centroid_original_space', np.single, (2,)),
    ('direction_original_space', np.single, (2,))
])

@dataclass
class EyesTrackerParamTracking(ParamTracking):
    eye_dyntresh_res: int = 20
    eye_size_lo_mm: float = 1.0
    eye_size_hi_mm: float = 10.0
    eye_thresh_lo: float = 0.0
    eye_thresh_hi: float = 1.0
    
    @property
    def eye_size_lo_px(self):
        return self.mm2px(self.eye_size_lo_mm)
    
    @property
    def eye_size_hi_px(self):
        return self.mm2px(self.eye_size_hi_mm)
    
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('empty', bool),
            ('centroid', np.float32, (2,)),
            ('heading_vector', np.float32, (2,)),
            ('origin',  np.int32, (2,)),
            ('left_eye', DTYPE_EYE),
            ('right_eye', DTYPE_EYE),
            ('mask',  np.bool_, self.crop_dimension_px[::-1]),
            ('image',  np.float32, self.crop_dimension_px[::-1]),
            ('image_fullres',  np.float32, self.source_crop_dimension_px[::-1]),
        ])
        return dt
    
@dataclass
class EyesTrackerParamOverlay:
    pix_per_mm: float = 40.0
    eye_len_mm: float = 0.25
    color_eye_left_BGR: tuple = (255, 255, 128)
    color_eye_right_BGR: tuple = (128, 255, 255)
    thickness: int = 1
    arrow_radius_mm: float = 0.1
    alpha: float = 0.5

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px
    
    @property
    def eye_len_px(self):
        return self.mm2px(self.eye_len_mm)
    
    @property
    def arrow_radius_px(self):
        return self.mm2px(self.arrow_radius_mm)
    
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
