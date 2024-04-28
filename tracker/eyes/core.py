from dataclasses import dataclass
from numpy.typing import NDArray
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
    direction: Optional[NDArray] = None
    angle: Optional[float] = None
    centroid: Optional[NDArray] = None

    def to_numpy(self, out: Optional[NDArray] = None) -> Optional[NDArray]:
        '''serialize to structured numpy array'''

        if out is not None:
            out['empty'] = self.direction is None
            out['direction'] = np.zeros((1,2), np.single) if self.direction is None else self.direction
            out['angle'] = np.zeros((1,), np.single) if self.angle is None else self.angle
            out['centroid'] = np.zeros((1,2), np.single) if self.centroid is None else self.centroid

        else:
            dt = np.dtype([
                ('empty', bool, (1,)),
                ('direction', np.single, (1,2)),
                ('angle', np.single, (1,)),
                ('centroid', np.single, (1,2))
            ])
                        
            arr = np.array(
                (
                    self.direction is None,
                    np.zeros((1,2), np.single) if self.direction is None else self.direction, 
                    np.zeros((1,), np.single) if self.angle is None else self.angle, 
                    np.zeros((1,2), np.single) if self.centroid is None else self.centroid
                ), 
                dtype=dt
            )
            return arr
    
    @classmethod
    def from_numpy(cls, array):
        instance = cls(
            direction = None if array['empty'][0] else array['direction'][0],
            angle = None if array['empty'][0] else array['angle'][0],
            centroid = None if array['empty'][0] else array['centroid'][0],
        )
        return instance

@dataclass    
class EyesTracking:
    im_eyes_shape: tuple
    im_eyes_fullres_shape: tuple
    mask: NDArray
    image: NDArray
    image_fullres: NDArray
    centroid: Optional[NDArray] = None
    origin: Optional[NDArray] = None
    left_eye: Eye = Eye()
    right_eye: Eye = Eye()
    
    def to_csv(self):
        '''export data as csv'''
        pass

    def to_numpy(self, out: Optional[NDArray] = None) -> Optional[NDArray]:
        '''serialize to fixed-size structured numpy array'''

        if out is not None:
            out['empty'] = self.centroid is None
            out['centroid'] = np.zeros((1,2), np.float32) if self.centroid is None else self.centroid
            out['origin'] = np.zeros((1,2), np.int32) if self.origin is None else self.origin
            if self.left_eye is not None:
                self.left_eye.to_numpy(out['left_eye'])
            if self.right_eye is not None:
                self.right_eye.to_numpy(out['right_eye'])
            out['mask'] = self.mask
            out['image'] = self.image
            out['image_fullres'] = self.image_fullres 

        else:
            left_eye = self.left_eye.to_numpy() if self.left_eye is not None else Eye().to_numpy()
            right_eye = self.right_eye.to_numpy() if self.right_eye is not None else Eye().to_numpy()

            dt = np.dtype([
                ('empty', bool, (1,)),
                ('centroid', np.float32, (1,2)),
                ('origin',  np.int32, (1,2)),
                ('left_eye',  left_eye.dtype, left_eye.shape),
                ('right_eye',  right_eye.dtype, right_eye.shape),
                ('mask',  np.bool_, self.im_eyes_shape),
                ('image',  np.float32, self.im_eyes_shape),
                ('image_fullres',  np.float32, self.im_eyes_fullres_shape),
            ])

            arr = np.array(
                (
                    self.centroid is None,
                    np.zeros((1,2), np.float32) if self.centroid is None else self.centroid,
                    np.zeros((1,2), np.int32) if self.origin is None else self.origin,
                    left_eye, 
                    right_eye,                
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
            im_eyes_shape = array['image'].shape,
            im_eyes_fullres_shape = array['image_fullres'].shape,
            mask = array['mask'],
            image = array['image'],
            image_fullres = array['image_fullres'],
            centroid = None if array['empty'][0] else array['centroid'][0],
            origin = None if array['empty'][0] else array['origin'][0],
            left_eye = Eye.from_numpy(array['left_eye']),
            right_eye = Eye.from_numpy(array['right_eye'])
        )
        return instance
    
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
