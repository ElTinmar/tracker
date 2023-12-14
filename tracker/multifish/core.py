from typing import Protocol, Optional, List
from numpy.typing import NDArray
from .core import Tracker, TrackingOverlay
from tracker.animal import AnimalTracking, AnimalOverlay, AnimalTracker
from tracker.body import BodyTracking, BodyOverlay, BodyTracker
from tracker.eyes import EyesTracking, EyesOverlay, EyesTracker
from tracker.tail import TailTracking, TailOverlay, TailTracker
from dataclasses import dataclass

class Accumulator(Protocol):
    def update(self):
        ...

class Assignment(Protocol):
    def update(self):
        ...
    
    def get_ID(self):
        ...

@dataclass
class MultiFishTracking:
    identities: NDArray
    indices: NDArray
    animals: AnimalTracking
    body: Optional[List[BodyTracking]]
    eyes: Optional[List[EyesTracking]]
    tail: Optional[List[TailTracking]]
    image: NDArray

    
    def to_csv(self):
        '''export data as csv'''
        pass

class MultiFishTracker(Tracker):

    def __init__(
            self, 
            assignment: Assignment,
            accumulator: Accumulator,
            animal: AnimalTracker,
            body: Optional[BodyTracker], 
            eyes: Optional[EyesTracker], 
            tail: Optional[TailTracker]
        ):
        self.assignment = assignment
        self.accumulator = accumulator
        self.animal = animal
        self.body = body
        self.eyes = eyes
        self.tail = tail

    
class MultiFishOverlay(TrackingOverlay):

    def __init__(
            self, 
            animal: AnimalOverlay,
            body: Optional[BodyOverlay], 
            eyes: Optional[EyesOverlay], 
            tail: Optional[TailOverlay]
        ) -> None:
        super().__init__()

        self.animal = animal
        self.body = body
        self.eyes = eyes
        self.tail = tail    
