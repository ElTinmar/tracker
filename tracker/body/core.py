from numpy.typing import NDArray, ArrayLike
from typing import Optional
import numpy as np
from dataclasses import dataclass
from tracker.core import Tracker, TrackingOverlay

@dataclass
class BodyTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    body_intensity: float = 0.1
    body_brightness: float = 0.2
    body_gamma: float = 1.0
    body_contrast: float = 1.0
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    min_body_size_mm: float = 10.0
    max_body_size_mm: float = 100.0
    min_body_length_mm: float = 2.0
    max_body_length_mm: float = 6.0
    min_body_width_mm: float = 1.0
    max_body_width_mm: float = 3.0

    def mm2px(self, val_mm):
        return int(val_mm * self.target_pix_per_mm) 

    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
    
    @property
    def min_body_size_px(self):
        return self.mm2px(self.min_body_size_mm)
    
    @property
    def max_body_size_px(self):
        return self.mm2px(self.max_body_size_mm) 
        
    @property
    def min_body_length_px(self):
        return self.mm2px(self.min_body_length_mm)
    
    @property
    def max_body_length_px(self):
        return self.mm2px(self.max_body_length_mm)

    @property
    def min_body_width_px(self):
        return self.mm2px(self.min_body_width_mm)
    
    @property
    def max_body_width_px(self):
        return self.mm2px(self.max_body_width_mm)

    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm)
    
    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm)

    def to_dict(self):
        res = {}
        res['pix_per_mm'] = self.pix_per_mm
        res['target_pix_per_mm'] = self.target_pix_per_mm
        res['body_intensity'] = self.body_intensity
        res['body_brightness'] = self.body_brightness
        res['body_gamma'] = self.body_gamma
        res['body_contrast'] = self.body_contrast
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['min_body_size_mm'] = self.min_body_size_mm
        res['max_body_size_mm'] = self.max_body_size_mm
        res['min_body_length_mm'] = self.min_body_length_mm
        res['max_body_length_mm'] = self.max_body_length_mm
        res['min_body_width_mm'] = self.min_body_width_mm
        res['max_body_width_mm'] = self.max_body_width_mm
        return res
    
@dataclass
class BodyTrackerParamOverlay:
    pix_per_mm: float = 40.0
    heading_len_mm: float = 1.25
    heading_color_BGR: tuple = (0,128,255)
    thickness: int = 2
    arrow_radius_mm: float = 0.1

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px

    @property
    def heading_len_px(self):
        return self.mm2px(self.heading_len_mm)

    @property
    def arrow_radius_px(self):
        return self.mm2px(self.arrow_radius_mm)        

class BodyTracking:

    def __init__(self,
            heading: Optional[NDArray] = None,
            centroid: Optional[NDArray] = None,
            angle_rad: Optional[float] = None,
            mask: Optional[NDArray] = None,
            image: Optional[NDArray] = None,
        ) -> None:
    
            self.heading = heading # 2x2 matrix, column 1 = fish heading, column 2 = fish right direction
            self.centroid = centroid # 1x2 vector. (x,y) coordinates of the fish centroid ~ swim bladder location
            self.angle_rad = angle_rad 
            self.mask = mask
            self.image = image 

    def to_csv(self):
        '''
        export data to csv
        '''
        pass    

    def to_numpy(
            self,           
            im_shape: Optional[ArrayLike] = None
        ) -> NDArray:
        '''serialize to fixed-size structured numpy array'''

        dt = np.dtype([
            ('heading', np.single, (2,2)),
            ('centroid',  np.single, (1,2)),
            ('angle_rad',  np.single, (1,)),
            ('mask',  np.uint8, im_shape),
            ('image',  np.uint8, im_shape),
        ])

        arr = np.array(
            (
                self.heading or np.zeros((2,2), np.single), 
                self.centroid or np.zeros((1,2), np.single),
                self.angle_rad or 0.0, 
                self.mask or np.zeros(im_shape, np.uint8), 
                self.image or np.zeros(im_shape, np.uint8)
            ), 
            dtype=dt
        )
        return arr

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
