from numpy.typing import NDArray
from typing import Optional, Tuple
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
    crop_dimension_mm: Tuple[float, float] = (5.5, 5.5) 

    def mm2px(self, val_mm):
        return int(val_mm * self.target_pix_per_mm) 

    def source_mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 
    
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
        res['crop_dimension_mm'] = self.crop_dimension_mm
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

@dataclass
class BodyTracking:
    im_body_shape: tuple
    im_body_fullres_shape: tuple
    mask: NDArray
    image: NDArray
    image_fullres: NDArray
    heading: Optional[NDArray] = None
    centroid: Optional[NDArray] = None
    angle_rad: Optional[float] = None

    def to_csv(self):
        '''
        export data to csv
        '''
        pass    

    def to_numpy(self, out: Optional[NDArray] = None) -> Optional[NDArray]:
        '''serialize to fixed-size structured numpy array'''

        if out is not None:
            out[0]['empty'] = self.heading is None
            out[0]['heading'] = np.zeros((2,2), np.float32) if self.heading is None else self.heading
            out[0]['centroid'] = np.zeros((1,2), np.float32) if self.centroid is None else self.centroid
            out[0]['angle_rad'] = 0.0 if self.angle_rad is None else self.angle_rad
            out[0]['mask'] = self.mask
            out[0]['image'] = self.image
            out[0]['image_fullres'] = self.image_fullres

        else:
            dt = np.dtype([
                ('empty', bool, (1,)),
                ('heading', np.float32, (2,2)),
                ('centroid', np.float32, (1,2)),
                ('angle_rad', np.float32, (1,)),
                ('mask', np.bool_, self.im_body_shape),
                ('image', np.float32, self.im_body_shape),
                ('image_fullres', np.float32, self.im_body_fullres_shape),
            ])

            arr = np.array(
                (
                    self.heading is None,
                    np.zeros((2,2), np.float32) if self.heading is None else self.heading, 
                    np.zeros((1,2), np.float32) if self.centroid is None else self.centroid,
                    0.0 if self.angle_rad is None else self.angle_rad, 
                    self.mask, 
                    self.image,
                    self.image_fullres
                ), 
                dtype=dt
            )
            return arr
    
    @classmethod
    def from_numpy(cls, array):
        instance = cls(
            im_body_shape = array['image'].shape,
            im_body_fullres_shape = array['image_fullres'].shape,
            heading = None if array['empty'][0] else array['heading'],
            centroid = None if array['empty'][0] else array['centroid'][0],
            angle_rad = None if array['empty'][0] else array['angle_rad'][0],
            mask = array['mask'],
            image = array['image'],
            image_fullres = array['image_fullres']
        )
        return instance

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
