from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from typing import Tuple
from tracker.core import Tracker, TrackingOverlay

DTYPE_EYE = np.dtype([
    ('direction', np.single, (1,2)),
    ('angle', np.single),
    ('centroid', np.single, (1,2)),
    ('centroid_original_space', np.single, (1,2)),
    ('direction_original_space', np.single, (1,2))
])

@dataclass
class EyesTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    eye_brightness: float = 0.2
    eye_gamma: float = 1.0
    eye_dyntresh_res: int = 20
    eye_contrast: float = 1.0
    eye_size_lo_mm: float = 1.0
    eye_size_hi_mm: float = 10.0
    eye_thresh_lo: float = 0.0
    eye_thresh_hi: float = 1.0
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    crop_dimension_mm: Tuple[float, float] = (1.2, 1.2) 
    crop_offset_mm: float = -0.3

    def mm2px(self, val_mm):
        return int(val_mm * self.target_pix_per_mm) 
    
    def source_mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 
    
    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
    
    @property
    def eye_size_lo_px(self):
        return self.mm2px(self.eye_size_lo_mm)
    
    @property
    def eye_size_hi_px(self):
        return self.mm2px(self.eye_size_hi_mm)
    
    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm)
    
    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm)
    
    @property
    def crop_dimension_px(self):
        # some video codec require height, width to be divisible by 2
        return (
            2* (self.mm2px(self.crop_dimension_mm[0])//2),
            2* (self.mm2px(self.crop_dimension_mm[1])//2)
        ) 

    @property
    def source_crop_dimension_px(self):
        # some video codec require height, width to be divisible by 2
        return (
            2* (self.source_mm2px(self.crop_dimension_mm[0])//2),
            2* (self.source_mm2px(self.crop_dimension_mm[1])//2)
        ) 
        
    @property
    def crop_offset_px(self):
        return self.source_mm2px(self.crop_offset_mm)

    @property
    def source_crop_offset_px(self):
        return self.source_mm2px(self.crop_offset_mm)
        
    def to_dict(self):
        res = {}
        res['pix_per_mm'] = self.pix_per_mm
        res['target_pix_per_mm'] = self.target_pix_per_mm
        res['eye_brightness'] = self.eye_brightness
        res['eye_gamma'] = self.eye_gamma
        res['eye_dyntresh_res'] = self.eye_dyntresh_res
        res['eye_contrast'] = self.eye_contrast
        res['eye_size_lo_mm'] = self.eye_size_lo_mm
        res['eye_size_hi_mm'] = self.eye_size_hi_mm
        res['eye_thresh_lo'] = self.eye_thresh_lo
        res['eye_thresh_hi'] = self.eye_thresh_hi
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['crop_dimension_mm'] = self.crop_dimension_mm
        res['crop_offset_mm'] = self.crop_offset_mm
        return res
    
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('empty', bool),
            ('centroid', np.float32, (1,2)),
            ('heading_vector', np.float32, (1,2)),
            ('origin',  np.int32, (1,2)),
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
