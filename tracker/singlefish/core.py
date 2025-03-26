from typing import Optional
from tracker.core import Tracker, TrackingOverlay
from tracker.animal import AnimalOverlay, AnimalTracker
from tracker.body import BodyOverlay, BodyTracker
from tracker.eyes import EyesOverlay, EyesTracker
from tracker.tail import TailOverlay, TailTracker
from dataclasses import dataclass
import numpy as np

@dataclass
class SingleFishTrackerParamTracking:
    animal: AnimalTracker
    body: Optional[BodyTracker]
    eyes: Optional[EyesTracker] 
    tail: Optional[TailTracker]

    @property
    def dtype(self) -> np.dtype:

        dt_list = [('animals', self.animal.tracking_param.dtype)]

        if self.body is not None:
            dt_list += [('body', self.body.tracking_param.dtype)]
        
        if self.eyes is not None:
            dt_list += [('eyes', self.eyes.tracking_param.dtype)]

        if self.tail is not None:
            dt_list += [('tail', self.tail.tracking_param.dtype)]
        
        dt = np.dtype(dt_list)
        return dt

    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)

@dataclass
class SingleFishTrackerParamOverlay:
    animal: AnimalOverlay
    body: Optional[BodyOverlay]
    eyes: Optional[EyesOverlay] 
    tail: Optional[TailOverlay]
        
class SingleFishTracker(Tracker):

    def __init__(
            self, 
            tracking_param: SingleFishTrackerParamTracking = SingleFishTrackerParamTracking()
        ):

        self.tracking_param = tracking_param

class SingleFishOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: SingleFishTrackerParamOverlay = SingleFishTrackerParamOverlay()
        ) -> None:
        super().__init__()

        self.overlay_param = overlay_param 
