from typing import Protocol, Optional, List
from numpy.typing import NDArray, ArrayLike
from tracker.core import Tracker, TrackingOverlay
from tracker.animal import AnimalTracking, AnimalOverlay, AnimalTracker, AnimalTracking
from tracker.body import BodyTracking, BodyOverlay, BodyTracker, BodyTracking
from tracker.eyes import EyesTracking, EyesOverlay, EyesTracker, EyesTracking
from tracker.tail import TailTracking, TailOverlay, TailTracker, TailTracking
from dataclasses import dataclass
import numpy as np

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

    def to_numpy(
            self,
            max_num_animals: int = 1,
            num_tail_pts: int = 20,
            num_tail_interp_pts: int = 40,
            im_shape: Optional[ArrayLike] = None,
            im_body_shape: Optional[ArrayLike] = None,
            im_eyes_shape: Optional[ArrayLike] = None,
            im_tail_shape: Optional[ArrayLike] = None
        ) -> NDArray:
        '''serialize to fixed-size structured numpy array'''

        animals = self.animals.to_numpy()
        bodies = [body.to_numpy() for body in self.body]
        eyes = [eyes.to_numpy() for eyes in self.eyes]
        tails = [tail.to_numpy() for tail in self.tail]

        # Pad identities, indices, bodies, eyes, tails 
        # up to max_num_animals with default empty value
        bodies += [BodyTracking().to_numpy(im_body_shape)] * (max_num_animals - len(bodies))
        eyes += [EyesTracking().to_numpy(im_eyes_shape)] * (max_num_animals - len(eyes))
        tails += [TailTracking().to_numpy(num_tail_pts, num_tail_interp_pts, im_tail_shape)] * (max_num_animals - len(eyes))
        
        dt = np.dtype([
            ('identities', self.identities.dtype, (max_num_animals,)),
            ('indices',  self.indices.dtype, (max_num_animals,)),
            ('animals',  animals.dtype, (1,)),
            ('bodies',  bodies[0].dtype, (max_num_animals,)),
            ('eyes',  eyes[0].dtype, (max_num_animals,)),
            ('tails',  tails[0].dtype, (max_num_animals,)),
            ('image',  np.uint8, im_shape)
        ])

        arr = np.array(
            (
                self.identities, 
                self.indices, 
                animals, 
                bodies, 
                eyes, 
                tails, 
                self.image
            ), 
            dtype=dt
        )
        return arr

class MultiFishTracker(Tracker):

    def __init__(
            self, 
            max_num_animals: int,
            assignment: Assignment,
            accumulator: Accumulator,
            animal: AnimalTracker,
            body: Optional[BodyTracker], 
            eyes: Optional[EyesTracker], 
            tail: Optional[TailTracker]
        ):
        self.max_num_animals = max_num_animals
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
