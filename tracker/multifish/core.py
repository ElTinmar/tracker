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
    max_num_animals: int
    image: NDArray
    animals: AnimalTracking
    body_tracked: bool = False
    eyes_tracked: bool = False
    tail_tracked: bool = False
    im_body_shape: Optional[tuple] = None
    im_eyes_shape: Optional[tuple] = None
    im_tail_shape: Optional[tuple] = None
    num_tail_pts: Optional[int] = None
    num_tail_interp_pts: Optional[int] = None 
    body: Optional[List[BodyTracking]] = None
    eyes: Optional[List[EyesTracking]] = None
    tail: Optional[List[TailTracking]] = None
    
    def to_csv(self):
        '''export data as csv'''
        pass

    def to_numpy(self) -> NDArray:
        '''serialize to fixed-size structured numpy array'''

        dt_tuples = []
        array_content = []

        dt_tuples.append(('max_num_animals', int, (1,)))
        array_content.append(self.max_num_animals)

        dt_tuples.append(('image', np.float32, self.image.shape))
        array_content.append(self.image)

        animals = self.animals.to_numpy() 
        dt_tuples.append(('animals', animals.dtype, (1,)))
        array_content.append(animals)

        dt_tuples.append(('body_tracked', np.bool_, (1,)))
        array_content.append(self.body_tracked)

        dt_tuples.append(('eyes_tracked', np.bool_, (1,)))
        array_content.append(self.eyes_tracked)

        dt_tuples.append(('tail_tracked', np.bool_, (1,)))
        array_content.append(self.tail_tracked)

        if self.body_tracked:
            bodies = [body.to_numpy() for id, body in self.body.items() if body is not None]
            bodies += [
                BodyTracking(
                    im_body_shape = self.im_body_shape, 
                    image = np.zeros(self.im_body_shape, dtype=np.float32), 
                    mask = np.zeros(self.im_body_shape, dtype=np.bool_)
                ).to_numpy()
            ] * (self.max_num_animals - len(bodies))
            dt_tuples.append(('bodies', bodies[0].dtype, (self.max_num_animals,)))
            array_content.append(bodies)
        
        if self.eyes_tracked:
            eyes = [eyes.to_numpy() for id, eyes in self.eyes.items() if eyes is not None]
            eyes += [
                EyesTracking(
                    im_eyes_shape = self.im_eyes_shape,
                    image = np.zeros(self.im_eyes_shape, dtype=np.float32),
                    mask = np.zeros(self.im_eyes_shape, dtype=np.bool_)
                ).to_numpy()
            ] * (self.max_num_animals - len(eyes))
            dt_tuples.append(('eyes', eyes[0].dtype, (self.max_num_animals,)))
            array_content.append(eyes)

        if self.tail_tracked:
            tails = [tail.to_numpy() for id, tail in self.tail.items() if tail is not None]
            tails += [
                TailTracking(
                    num_tail_pts = self.num_tail_pts,
                    num_tail_interp_pts = self.num_tail_interp_pts,
                    im_tail_shape = self.im_tail_shape,
                    image = np.zeros(self.im_tail_shape, dtype=np.float32),
                ).to_numpy()
            ] * (self.max_num_animals - len(tails))
            dt_tuples.append(('tails', tails[0].dtype, (self.max_num_animals,)))
            array_content.append(tails)

        arr = np.array(tuple(array_content), dtype= np.dtype(dt_tuples, align=True))
        return arr

    @classmethod
    def from_numpy(cls, array):
        instance = cls(
            max_num_animals =
            image =
            animals =
            body_tracked =
            eyes_tracked =
            tail_tracked =
            im_body_shape =
            im_eyes_shape =
            im_tail_shape =
            num_tail_pts =
            num_tail_interp_pts = 
            body =
            eyes =
            tail =
        )
        return instance
    
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
