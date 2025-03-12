from dataclasses import dataclass
import numpy as np
from tracker.core import Tracker, TrackingOverlay, ParamTracking

@dataclass
class TailTrackerParamTracking(ParamTracking):
    arc_angle_deg: float = 120.0
    n_tail_points: int = 12
    n_pts_arc: int = 20
    n_pts_interp: int = 40
    tail_length_mm: float = 2.6
    ball_radius_mm: float = 0.05 
     
    @property
    def tail_length_px(self) -> int:
        return self.target_mm2px(self.tail_length_mm)
    
    @property
    def ball_radius_px(self) -> int:
        return self.target_mm2px(self.ball_radius_mm) 
    
    @property
    def dtype(self) -> np.dtype:
        dt = np.dtype([
            ('num_pts', int),
            ('num_interp_pts', int),
            ('centroid', np.float32, (2,)),
            ('skeleton_resized',  np.float32, (self.n_tail_points,2)),
            ('skeleton_cropped',  np.float32, (self.n_tail_points,2)),
            ('skeleton_input',  np.float32, (self.n_tail_points,2)),
            ('skeleton_global',  np.float32, (self.n_tail_points,2)),
            ('skeleton_interp_resized',  np.float32, (self.n_pts_interp,2)),
            ('skeleton_interp_cropped',  np.float32, (self.n_pts_interp,2)),
            ('skeleton_interp_input',  np.float32, (self.n_pts_interp,2)),
            ('skeleton_interp_global',  np.float32, (self.n_pts_interp,2)),
            ('image_processed',  np.float32, self.resized_dimension_px[::-1]),
            ('image_cropped',  np.float32, self.crop_dimension_px[::-1]),
            ('pix_per_mm_global', np.float32),
            ('pix_per_mm_input', np.float32),
            ('pix_per_mm_cropped', np.float32),
            ('pix_per_mm_resized', np.float32),
        ])
        return dt

@dataclass
class TailTrackerParamOverlay:
    color_BGR: tuple = (255, 128, 128)
    thickness: int = 1
    ball_radius_mm: float = 0.1 
    alpha: float = 0.5

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
