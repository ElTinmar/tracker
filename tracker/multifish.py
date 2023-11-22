import numpy as np
import cv2
from typing import Protocol, Optional, Dict
from numpy.typing import NDArray
from image_tools import enhance, imrotate, im2rgb
from geometry import Affine2DTransformation
from .tracker import Tracker

class Accumulator(Protocol):
    def update(self):
        ...

class Assignment(Protocol):
    def update(self):
        ...
    
    def get_ID(self):
        ...

class MultiFish(Tracker):

    def __init__(
            self, 
            assignment: Assignment,
            accumulator: Accumulator,
            animal_tracker: Tracker,
            body_tracker: Optional[Tracker], 
            eyes_tracker: Optional[Tracker], 
            tail_tracker: Optional[Tracker]
        ):
        self.assignment = assignment
        self.accumulator = accumulator
        self.animal_tracker = animal_tracker
        self.body_tracker = body_tracker
        self.eyes_tracker = eyes_tracker
        self.tail_tracker = tail_tracker
        
    def track(self, image: NDArray, centroid: Optional[NDArray] = None):

        if (image is None) or (image.size == 0):
            return None
        
        # restrain image between 0 and 1
        image = enhance(image)

        # get animal centroids (only crude location is necessary)
        animals = self.animal_tracker.track(image)
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
            if self.body_tracker is not None:

                # get more precise centroid and orientation of the animals
                
                body[id] = self.body_tracker.track(image_cropped, centroi=offset)
                if body[id] is not None:
                    
                    # rotate the animal so that it's vertical head up
                    image_rot, centroid_rot = imrotate(
                        image_cropped, 
                        body[id].centroid[0], body[id].centroid[1], 
                        np.rad2deg(body[id].angle_rad)
                    )

                    # track eyes 
                    if self.eyes_tracker is not None:
                        eyes[id] = self.eyes_tracker.track(image_rot, centroid=centroid_rot)

                    # track tail
                    if self.tail_tracker is not None:
                        tail[id] = self.tail_tracker.track(image_rot, centroid=centroid_rot)

                # compute additional features based on tracking
                if self.accumulator is not None:
                    self.accumulator.update(id,body[id],eyes[id],tail[id])

        # save tracking results in a dict and return
        res = {
            'identities': identities, 
            'indices': to_keep,
            'animals': animals,
            'body': body,
            'eyes': eyes,
            'tail': tail,
            'image': (255*image).astype(np.uint8)
        }
        return res 

    def overlay(
            self, 
            image: NDArray, 
            tracking: Optional[Dict], 
            transformation_matrix: NDArray
        ) -> NDArray:

        if tracking is not None:

            overlay = im2rgb(image)

            # loop over animals
            for idx, id in zip(tracking['indices'], tracking['identities']):
                if tracking['animals'] is not None:

                    # overlay animal bounding boxes
                    overlay = self.animal_tracker.overlay(overlay, tracking['animals'])
                    
                    # translate according to animal position 
                    bbox_bottomleft = tracking['animals'].bounding_boxes[idx,:2]

                    if (self.body_tracker is not None)  and (tracking['body'][id] is not None):

                        # rotate according to animal orientation 
                        angle = tracking['body'][id].angle_rad
                        rotation = rotation_matrix(np.rad2deg(angle))[:2,:2]
                        
                        # overlay body
                        overlay = self.body_tracker.overlay(overlay, tracking['body'][id], bbox_bottomleft)
                        
                        # overlay eyes
                        if (self.eyes_tracker is not None)  and (tracking['eyes'][id]is not None):
                            offset_eye_ROI = bbox_bottomleft + tracking['body'][id].centroid 
                            overlay = self.eyes_tracker.overlay(overlay, tracking['eyes'][id], offset_eye_ROI, rotation)
                        
                        # overlay tail
                        if (self.tail_tracker is not None)  and (tracking['tail'][id] is not None):
                            offset_tail_ROI = bbox_bottomleft + tracking['body'][id].centroid 
                            overlay = self.tail_tracker.overlay(overlay, tracking['tail'][id], offset_tail_ROI, rotation)

                # show ID
                cv2.putText(overlay, str(id), bbox_bottomleft.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
            
            return overlay
