from typing import Optional
from tracker.core import Tracker, TrackingOverlay
from tracker.animal import AnimalOverlay, AnimalTracker, AnimalOverlay_opencv, AnimalTracker_CPU
from tracker.body import BodyOverlay, BodyTracker, BodyOverlay_opencv, BodyTracker_CPU
from tracker.eyes import EyesOverlay, EyesTracker, EyesOverlay_opencv, EyesTracker_CPU
from tracker.tail import TailOverlay, TailTracker, TailOverlay_opencv, TailTracker_CPU
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MultiFishTrackerParamTracking:
    animal: AnimalTracker = field(default_factory=AnimalTracker_CPU)
    body: Optional[BodyTracker] = field(default_factory=BodyTracker_CPU)
    eyes: Optional[EyesTracker] = field(default_factory=EyesTracker_CPU)
    tail: Optional[TailTracker] = field(default_factory=TailTracker_CPU)

    @property
    def dtype(self) -> np.dtype:

        dt_list = [
            ('success', np.bool_),
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
    animal: AnimalOverlay = field(default_factory=AnimalOverlay_opencv)
    body: Optional[BodyOverlay] = field(default_factory=BodyOverlay_opencv)
    eyes: Optional[EyesOverlay] = field(default_factory=EyesOverlay_opencv)
    tail: Optional[TailOverlay] = field(default_factory=TailOverlay_opencv)
        
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
