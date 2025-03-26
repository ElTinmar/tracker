from typing import Optional
from tracker.core import Tracker, TrackingOverlay
from tracker.animal import AnimalOverlay, AnimalTracker
from tracker.body import BodyOverlay, BodyTracker
from tracker.eyes import EyesOverlay, EyesTracker
from tracker.tail import TailOverlay, TailTracker
from dataclasses import dataclass
import numpy as np

@dataclass
class MultiFishTrackerParamTracking:
    animal: AnimalTracker = AnimalTracker()
    body: Optional[BodyTracker] = BodyTracker()
    eyes: Optional[EyesTracker] = EyesTracker()
    tail: Optional[TailTracker] = TailTracker()

    @property
    def dtype(self) -> np.dtype:

        dt_list = [
            ('animals', self.animal.tracking_param.dtype),
        ]

        if self.body is not None:
            dt_list += [(
                'body', 
                self.body.tracking_param.dtype, 
                (self.animal.tracking_param.num_animals,)
            )]
        
        if self.eyes is not None:
            dt_list += [(
                'eyes', 
                self.eyes.tracking_param.dtype,
                (self.animal.tracking_param.num_animals,)
            )]

        if self.tail is not None:
            dt_list += [(
                'tail', 
                self.tail.tracking_param.dtype,
                (self.animal.tracking_param.num_animals,)
            )]
        
        dt = np.dtype(dt_list)
        return dt

    @property
    def failed(self):
        return np.zeros((), dtype=self.dtype)

@dataclass
class MultiFishTrackerParamOverlay:
    animal: AnimalOverlay = AnimalOverlay()
    body: Optional[BodyOverlay] = BodyOverlay()
    eyes: Optional[EyesOverlay] = EyesOverlay()
    tail: Optional[TailOverlay] = TailOverlay()
        
class MultiFishTracker(Tracker):

    def __init__(
            self, 
            tracking_param: MultiFishTrackerParamTracking = MultiFishTrackerParamTracking() 
        ):

        self.tracking_param = tracking_param

class MultiFishOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: MultiFishTrackerParamOverlay = MultiFishTrackerParamOverlay() 
        ) -> None:
        super().__init__()

        self.overlay_param = overlay_param 
