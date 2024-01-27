from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike
import numpy as np
from typing import Tuple, Optional
from tracker.core import Tracker, TrackingOverlay

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
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    crop_dimension_mm: Tuple[float, float] = (1.2, 1.2) 
    crop_offset_mm: float = -0.3

    def mm2px(self, val_mm):
        return int(val_mm * self.target_pix_per_mm) 
    
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
    def crop_offset_px(self):
        return self.mm2px(self.crop_offset_mm)
    
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
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['crop_dimension_mm'] = self.crop_dimension_mm
        res['crop_offset_mm'] = self.crop_offset_mm
        return res
    
@dataclass
class EyesTrackerParamOverlay:
    pix_per_mm: float = 40.0
    eye_len_mm: float = 0.25
    color_eye_left_BGR: tuple = (255, 255, 128)
    color_eye_right_BGR: tuple = (128, 255, 255)
    thickness: int = 2
    arrow_radius_mm: float = 0.1


    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px
    
    @property
    def eye_len_px(self):
        return self.mm2px(self.eye_len_mm)
    
    @property
    def arrow_radius_px(self):
        return self.mm2px(self.arrow_radius_mm)

@dataclass
class Eye:
    direction: Optional[NDArray]
    angle: Optional[float]
    centroid: Optional[NDArray]

    def to_numpy(self) -> NDArray:
        '''serialize to structured numpy array'''

        dt = np.dtype([
            ('direction', np.single, (1,2)),
            ('angle', np.single, (1,)),
            ('centroid', np.single, (1,2))
        ])

        arr = np.array(
            (
                self.direction or np.zeros((1,2), np.single), 
                self.angle or np.zeros((1,), np.single), 
                self.centroid or np.zeros((1,2), np.single)
            ), 
            dtype=dt
        )
        return arr

class EyesTracking:
    def __init__(
            self,
            mask: Optional[NDArray],
            image: Optional[NDArray],
            centroid: Optional[NDArray],
            offset: Optional[NDArray],
            left_eye: Eye = Eye(None,None,None),
            right_eye: Eye = Eye(None,None,None),
        ) -> None:
    
        self.centroid = centroid
        self.offset = offset # position of centroid in cropped image
        self.left_eye =  left_eye
        self.right_eye = right_eye
        self.mask = mask
        self.image = image
    
    def to_csv(self):
        '''export data as csv'''
        pass

    def to_numpy(
            self,
            im_shape: Optional[ArrayLike] = None
        ) -> NDArray:
        '''serialize to fixed-size structured numpy array'''

        left_eye = self.left_eye.to_numpy()
        right_eye = self.right_eye.to_numpy()

        dt = np.dtype([
            ('centroid', np.single, (1,2)),
            ('offset',  np.single, (1,2)),
            ('left_eye',  left_eye.dtype, left_eye.shape),
            ('right_eye',  right_eye.dtype, right_eye.shape),
            ('mask',  np.uint8, im_shape),
            ('image',  np.uint8, im_shape),
        ])

        arr = np.array(
            (
                self.centroid or np.zeros((1,2), np.single),
                self.offset or np.zeros((1,2), np.single),
                left_eye, 
                right_eye,                
                self.mask or np.zeros(im_shape, np.uint8), 
                self.image or np.zeros(im_shape, np.uint8)
            ), 
            dtype=dt
        )
        return arr

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
