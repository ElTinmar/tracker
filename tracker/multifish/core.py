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


@dataclass
class MultiFishTracking:
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
            im_animal_shape: Optional[ArrayLike] = None,
            im_body_shape: Optional[ArrayLike] = None,
            im_eyes_shape: Optional[ArrayLike] = None,
            im_tail_shape: Optional[ArrayLike] = None
        ) -> NDArray:
        '''serialize to fixed-size structured numpy array'''

        #
        animals = self.animals.to_numpy(max_num_animals=max_num_animals, im_shape=im_animal_shape) 
        bodies = [body.to_numpy(im_shape=im_body_shape) for id, body in self.body.items() if body is not None]
        eyes = [eyes.to_numpy(im_shape=im_eyes_shape) for id, eyes in self.eyes.items() if eyes is not None]
        tails = [tail.to_numpy(im_shape=im_tail_shape,num_tail_pts=num_tail_pts,num_tailinterp_pts=num_tail_interp_pts) for id, tail in self.tail.items() if tail is not None]

        # pad missing data
        bodies += [BodyTracking().to_numpy(im_body_shape)] * (max_num_animals - len(bodies))
        eyes += [EyesTracking().to_numpy(im_eyes_shape)] * (max_num_animals - len(eyes))
        tails += [TailTracking().to_numpy(num_tail_pts, num_tail_interp_pts, im_tail_shape)] * (max_num_animals - len(tails))
        
        dt = np.dtype([
            ('animals', animals.dtype, (1,)),
            ('bodies', bodies[0].dtype, (max_num_animals,)),
            ('eyes', eyes[0].dtype, (max_num_animals,)),
            ('tails', tails[0].dtype, (max_num_animals,)),
            ('image', np.float32, im_shape)
        ])

        arr = np.array(
            (
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
            accumulator: Accumulator,
            animal: AnimalTracker,
            body: Optional[BodyTracker], 
            eyes: Optional[EyesTracker], 
            tail: Optional[TailTracker]
        ):
        self.max_num_animals = max_num_animals
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
