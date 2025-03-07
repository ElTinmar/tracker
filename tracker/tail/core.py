from dataclasses import dataclass
import numpy as np
from tracker.core import Tracker, TrackingOverlay, ParamTracking

@dataclass
class TailTrackerParamTracking(ParamTracking):
    arc_angle_deg: float = 120.0
    n_points: int = 12
    n_pts_arc: int = 20
    n_pts_interp: int = 40
    length_mm: float = 2.6
    ball_radius_mm: float = 0.05 
     
    @property
    def length_px(self):
        return self.mm2px(self.length_mm)
    
    @property
    def ball_radius_px(self):
        return self.mm2px(self.ball_radius_mm) 
    
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('empty', bool),
            ('num_pts', int),
            ('num_interp_pts', int),
            ('centroid', np.float32, (2,)),
            ('origin',  np.float32, (2,)),
            ('skeleton',  np.float32, (self.n_points,2)),
            ('skeleton_interp',  np.float32, (self.n_pts_interp,2)),
            ('image',  np.float32, self.resized_dimension_px[::-1]),
            ('image_fullres',  np.float32, self.crop_dimension_px[::-1])
        ])
        return dt

@dataclass
class TailTrackerParamOverlay:
    pix_per_mm: float = 40
    color_BGR: tuple = (255, 128, 128)
    thickness: int = 1
    ball_radius_mm: float = 0.1 
    alpha: float = 0.5

    def mm2px(self, val_mm: float) -> int:
        return int(val_mm * self.pix_per_mm) 

    @property
    def ball_radius_px(self):
        return self.mm2px(self.ball_radius_mm) 

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
