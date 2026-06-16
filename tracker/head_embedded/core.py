from abc import ABC, abstractmethod
from tracker.core import Tracker, TrackingOverlay
from tracker.tail import TailOverlay, TailTracker, TailOverlay_opencv, TailTracker_CPU
from .position_predictor import PositionPredictor
from .lighthill import LighthillPredictor
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HeadEmbedded_ParamTracking:
    tail: TailTracker = field(default_factory=TailTracker_CPU)
    position_estimator: PositionPredictor = field(default_factory=LighthillPredictor)
    centroid_x: float = 0.0
    centroid_y: float = 0.0 
    heading_angle_rad: float = 0.0

    @property
    def dtype(self) -> np.dtype:

        dt_list = [
            ('success', np.bool_),
            ('predicted_x', np.float32),
            ('predicted_y', np.float32),
            ('predicted_theta', np.float32),
            ('tail', self.tail.tracking_param.dtype)
        ]
        
        dt = np.dtype(dt_list)
        return dt

    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)

@dataclass
class HeadEmbedded_ParamOverlay:
    tail: TailOverlay = field(default_factory=TailOverlay_opencv)
        
class HeadEmbeddedTracker(Tracker):

    def __init__(
            self, 
            tracking_param: HeadEmbedded_ParamTracking = HeadEmbedded_ParamTracking()
        ):

        self.tracking_param = tracking_param

class HeadEmbeddedOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: HeadEmbedded_ParamOverlay = HeadEmbedded_ParamOverlay()
        ) -> None:
        super().__init__()

        self.overlay_param = overlay_param 

