from typing import Protocol, Optional, Dict
from numpy.typing import NDArray
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
    body: Optional[Dict[int, BodyTracking]] = None
    eyes: Optional[Dict[int, EyesTracking]] = None
    tail: Optional[Dict[int, TailTracking]] = None
    
    def to_csv(self):
        '''export data as csv'''
        pass

    def to_numpy(self) -> NDArray:
        # TODO duplicated code, write functions
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

        # TODO: this is mostly repeated 3 times, write a function 
        if self.body_tracked:
            bodies = [body.to_numpy() for id, body in self.body.items() if body is not None]
            pad_len = (self.max_num_animals - len(bodies))
            bodies += [
                BodyTracking(
                    im_body_shape = self.im_body_shape, 
                    image = np.zeros(self.im_body_shape, dtype=np.float32), 
                    mask = np.zeros(self.im_body_shape, dtype=np.bool_)
                ).to_numpy()
            ] * (self.max_num_animals - len(bodies))
            dt_tuples.append(('bodies', bodies[0].dtype, (self.max_num_animals,)))
            array_content.append(bodies)

            bodies_ids = [id for id, body in self.body.items() if body is not None]
            bodies_ids += [-1] * pad_len
            dt_tuples.append(('bodies_id', int, (self.max_num_animals,)))
            array_content.append(bodies_ids)
        
        if self.eyes_tracked:
            eyes = [eyes.to_numpy() for id, eyes in self.eyes.items() if eyes is not None]
            pad_len = (self.max_num_animals - len(eyes))
            eyes += [
                EyesTracking(
                    im_eyes_shape = self.im_eyes_shape,
                    image = np.zeros(self.im_eyes_shape, dtype=np.float32),
                    mask = np.zeros(self.im_eyes_shape, dtype=np.bool_)
                ).to_numpy()
            ] * pad_len
            dt_tuples.append(('eyes', eyes[0].dtype, (self.max_num_animals,)))
            array_content.append(eyes)

            eyes_ids = [id for id, eyes in self.eyes.items() if eyes is not None]
            eyes_ids += [-1] * pad_len
            dt_tuples.append(('eyes_id', int, (self.max_num_animals,)))
            array_content.append(eyes_ids)

        if self.tail_tracked:
            tails = [tail.to_numpy() for id, tail in self.tail.items() if tail is not None]
            pad_len = (self.max_num_animals - len(tails))
            tails += [
                TailTracking(
                    num_tail_pts = self.num_tail_pts,
                    num_tail_interp_pts = self.num_tail_interp_pts,
                    im_tail_shape = self.im_tail_shape,
                    image = np.zeros(self.im_tail_shape, dtype=np.float32),
                ).to_numpy()
            ] * pad_len
            dt_tuples.append(('tails', tails[0].dtype, (self.max_num_animals,)))
            array_content.append(tails)

            tails_ids = [id for id, tail in self.tail.items() if tail is not None]
            tails_ids += [-1] * pad_len
            dt_tuples.append(('tails_id', int, (self.max_num_animals,)))
            array_content.append(tails_ids)

        arr = np.array(tuple(array_content), dtype= np.dtype(dt_tuples, align=True))
        return arr

    @classmethod
    def from_numpy(cls, array):
        print(array['bodies_id'],array['bodies'])
        instance = cls(
            max_num_animals = array['max_num_animals'][0],
            image = array['image'],
            animals = AnimalTracking.from_numpy(array['animals']),
            body_tracked = array['body_tracked'][0],
            eyes_tracked = array['eyes_tracked'][0],
            tail_tracked = array['tail_tracked'][0],
            body = None if not array['body_tracked'][0] else {id: BodyTracking.from_numpy(tracking) for id, tracking in zip(array['bodies_id'],array['bodies'])},
            eyes = None if not array['eyes_tracked'][0] else {id: EyesTracking.from_numpy(tracking) for id, tracking in zip(array['eyes_id'],array['eyes'])},
            tail = None if not array['tail_tracked'][0] else {id: TailTracking.from_numpy(tracking) for id, tracking in zip(array['tails_id'],array['tails'])}
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
