from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike
import numpy as np
from typing import Tuple, Optional
from tracker.core import Tracker, TrackingOverlay

@dataclass
class TailTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    tail_contrast: float = 1.0,
    tail_gamma: float = 1.0,
    tail_brightness: float = 0.2,
    arc_angle_deg: float = 120.0
    n_tail_points: int = 12
    n_pts_arc: int = 20
    n_pts_interp: int = 40
    tail_length_mm: float = 2.6
    dist_swim_bladder_mm: float = 0.4
    blur_sz_mm: float = 0.10
    median_filter_sz_mm: float = 0.110
    crop_dimension_mm: Tuple[float, float] = (1.5, 1.5) 
    crop_offset_tail_mm: float = 2.25
    ball_radius_mm: float = 0.05 
    
    def mm2px(self, val_mm: float) -> int:
        return int(val_mm * self.target_pix_per_mm) 

    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
       
    @property
    def tail_length_px(self):
        return self.mm2px(self.tail_length_mm)
    
    @property
    def dist_swim_bladder_px(self):
        return self.mm2px(self.dist_swim_bladder_mm)

    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm) 

    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm) 

    @property
    def crop_offset_tail_px(self):
        return self.mm2px(self.crop_offset_tail_mm) 

    @property
    def ball_radius_px(self):
        return self.mm2px(self.ball_radius_mm) 
    
    @property
    def crop_dimension_px(self):
        # some video codec require height, width to be divisible by 2
        return (
            2 * (self.mm2px(self.crop_dimension_mm[0])//2),
            2 * (self.mm2px(self.crop_dimension_mm[1])//2)
        ) 

    def to_dict(self):
        res = {}
        res['pix_per_mm'] = self.pix_per_mm
        res['target_pix_per_mm'] = self.target_pix_per_mm
        res['tail_contrast'] = self.tail_contrast
        res['tail_gamma'] = self.tail_gamma
        res['tail_brightness'] = self.tail_brightness
        res['arc_angle_deg'] = self.arc_angle_deg
        res['n_tail_points'] = self.n_tail_points
        res['n_pts_arc'] = self.n_pts_arc
        res['n_pts_interp'] = self.n_pts_interp
        res['tail_length_mm'] = self.tail_length_mm
        res['dist_swim_bladder_mm'] = self.dist_swim_bladder_mm
        res['blur_sz_mm'] = self.blur_sz_mm
        res['median_filter_sz_mm'] = self.median_filter_sz_mm
        res['crop_dimension_mm'] = self.crop_dimension_mm
        res['crop_offset_tail_mm'] = self.crop_offset_tail_mm
        res['ball_radius_mm'] = self.ball_radius_mm
        return res
    

@dataclass
class TailTrackerParamOverlay:
    pix_per_mm: float = 40
    color_tail_BGR: tuple = (255, 128, 128)
    thickness: int = 2
    ball_radius_mm: float = 0.1 

    def mm2px(self, val_mm: float) -> int:
        return int(val_mm * self.pix_per_mm) 

    @property
    def ball_radius_px(self):
        return self.mm2px(self.ball_radius_mm) 
        
class TailTracking:
    def __init__(
            self,
            num_tail_pts: int,
            num_tail_interp_pts: int,
            im_tail_shape: tuple,
            centroid: Optional[NDArray] = None,
            offset: Optional[NDArray] = None,
            skeleton: Optional[NDArray] = None,
            skeleton_interp: Optional[NDArray] = None,
            image: Optional[NDArray] = None
        ) -> None:
                
            self.num_tail_pts = num_tail_pts
            self.num_tail_interp_pts = num_tail_interp_pts
            self.im_tail_shape = im_tail_shape
            self.centroid = centroid 
            self.offset = offset
            self.skeleton = skeleton
            self.skeleton_interp = skeleton_interp
            self.image = image

    def to_csv(self):
        '''export data as csv'''
        pass

    def to_numpy(self) -> NDArray:
        '''serialize to fixed-size structured numpy array'''
        
        dt = np.dtype([
            ('centroid', np.float32, (1,2)),
            ('offset',  np.float32, (1,2)),
            ('skeleton',  np.float32, (self.num_tail_pts,2)),
            ('skeleton_interp',  np.float32, (self.num_tail_interp_pts,2)),
            ('image',  np.float32, self.im_tail_shape)
        ])

        arr = np.array(
            (
                np.zeros((1,2), np.float32) if self.centroid is None else self.centroid, 
                np.zeros((1,2), np.float32) if self.offset is None else self.offset,
                np.zeros((self.num_tail_pts,2), np.float32) if self.skeleton is None else self.skeleton,
                np.zeros((self.num_tail_interp_pts,2), np.float32) if self.skeleton_interp is None else self.skeleton_interp,
                np.zeros(self.im_tail_shape, np.float32) if self.image is None else self.image
            ), 
            dtype=dt
        )
        return arr

class TailTracker(Tracker):

    def __init__(
            self, 
            tracking_param: TailTrackerParamTracking, 
        ) -> None:

        self.tracking_param = tracking_param

class TailOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: TailTrackerParamOverlay
        ) -> None:

        self.overlay_param = overlay_param
