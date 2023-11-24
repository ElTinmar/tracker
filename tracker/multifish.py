import numpy as np
import cv2
from typing import Protocol, Optional, List
from numpy.typing import NDArray
from image_tools import enhance, imrotate, im2rgb
from geometry import Affine2DTransform
from .core import Tracker, TrackingOverlay
from .animal import AnimalTracking
from .body import BodyTracking
from .eyes import EyesTracking
from .tail import TailTracking 
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
            animal: Tracker,
            body: Optional[Tracker], 
            eyes: Optional[Tracker], 
            tail: Optional[Tracker]
        ):
        self.assignment = assignment
        self.accumulator = accumulator
        self.animal = animal
        self.body = body
        self.eyes = eyes
        self.tail = tail
        
    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[MultiFishTracking]:

        if (image is None) or (image.size == 0):
            return None
        
        # restrain image between 0 and 1
        image = enhance(image)

        # get animal centroids (only crude location is necessary)
        animals = self.animal.track(image)
        centroids = animals.centroids

        # if nothing was detected at that stage, stop here
        if centroids.size == 0:
            return
        
        # assign identities to animals 
        self.assignment.update(centroids)
        identities = self.assignment.get_ID()
        to_keep = self.assignment.get_kept_centroids()        
        data = np.hstack(
            (identities[np.newaxis].T, 
             animals.bb_centroids[to_keep,:], 
             animals.bounding_boxes[to_keep,:])
        ) 

        # loop over animals
        body = {}
        eyes = {}
        tail = {}
        for (id, bb_x, bb_y, left, bottom, right, top) in data.astype(np.int64): 
            eyes[id] = None
            tail[id] = None
            body[id] = None

            # crop each animal's bounding box
            image_cropped = image[bottom:top, left:right] 
            offset = np.array([bb_x, bb_y])
            if self.body is not None:

                # get more precise centroid and orientation of the animals
                
                body[id] = self.body.track(image_cropped, centroid=offset)
                if (body[id] is not None) and (body[id].centroid is not None):
                    
                    # rotate the animal so that it's vertical head up
                    image_rot, centroid_rot = imrotate(
                        image_cropped, 
                        body[id].centroid[0], body[id].centroid[1], 
                        np.rad2deg(body[id].angle_rad)
                    )

                    # track eyes 
                    if self.eyes is not None:
                        eyes[id] = self.eyes.track(image_rot, centroid=centroid_rot)

                    # track tail
                    if self.tail is not None:
                        tail[id] = self.tail.track(image_rot, centroid=centroid_rot)

                # compute additional features based on tracking
                if self.accumulator is not None:
                    self.accumulator.update(id,body[id],eyes[id],tail[id])

        # save tracking results in a dict and return
        res = MultiFishTracking(
            identities =  identities, 
            indices = to_keep,
            animals = animals,
            body = body,
            eyes = eyes,
            tail =  tail,
            image = (255*image).astype(np.uint8)
        )
        
        return res 

class MultiFishOverlay(TrackingOverlay):

    def __init__(
            self, 
            animal: TrackingOverlay,
            body: Optional[TrackingOverlay], 
            eyes: Optional[TrackingOverlay], 
            tail: Optional[TrackingOverlay]
        ) -> None:
        super().__init__()

        self.animal = animal
        self.body = body
        self.eyes = eyes
        self.tail = tail    

    def overlay(
            self, 
            image: NDArray, 
            tracking: Optional[MultiFishTracking], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> NDArray:
        '''
        There are 3 different coordinate systems:
        - 1. image coordinates: the whole image, origin = image topleft
        - 2. bbox coordinates: cropped image of each animal, origin = bounding box top left coordinates 
        - 3. fish coordinates: fish centric coordinates, rotation = fish heading, origin = fish centroid
        '''

        if tracking is not None:

            overlay = im2rgb(image)

            # loop over animals
            for idx, id in zip(tracking.indices, tracking.identities):

                if tracking.animals is not None:

                    # overlay animal bounding boxes, coord system 1.
                    overlay = self.animal.overlay(overlay, tracking.animals)
                    
                    # transformation matrix from coord system 1. to coord system 2., just a translation  
                    tx_bbox = tracking.animals.bounding_boxes[idx,0]
                    ty_bbox = tracking.animals.bounding_boxes[idx,1]
                    translation_bbox = Affine2DTransform.translation(tx_bbox,ty_bbox)

                    if (self.body is not None) and (tracking.body[id] is not None) and (tracking.body[id].centroid is not None):

                        # overlay body, coord. system 2.
                        overlay = self.body.overlay(
                            overlay, 
                            tracking.body[id], 
                            translation_bbox
                        )

                        # transformation matrix from coord system 1. to coord system 3., rotation + translation
                        angle = tracking.body[id].angle_rad
                        rotation = Affine2DTransform.rotation(np.rad2deg(angle))
                        tx, ty = tracking.body[id].centroid 
                        transformation = rotation @ Affine2DTransform.translation(tx, ty) @ translation_bbox
                        
                        # overlay eyes, coord system 3.
                        if (self.eyes is not None)  and (tracking.eyes[id]is not None):
                            overlay = self.eyes.overlay(
                                overlay, 
                                tracking.eyes[id], 
                                transformation
                            )
                        
                        # overlay tail, coord system 3.
                        if (self.tail is not None)  and (tracking.tail[id] is not None):
                            overlay = self.tail.overlay(
                                overlay, 
                                tracking.tail[id], 
                                transformation
                            )

                # show ID, coord. system 1.
                cv2.putText(overlay, str(id), (int(tx_bbox), int(ty_bbox)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
            
            return overlay
