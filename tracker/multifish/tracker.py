import numpy as np
from typing import Optional
from numpy.typing import NDArray
from image_tools import imrotate, im2uint8
from .core import MultiFishTracker, MultiFishTracking

class MultiFishTracker_CPU(MultiFishTracker):

    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[MultiFishTracking]:

        if (image is None) or (image.size == 0):
            return None

        # get animal centroids (only crude location is necessary)
        animals = self.animal.track(image)

        # if nothing was detected at that stage, stop here
        if animals.identities.size == 0:
            res = MultiFishTracking(
                identities =  None, 
                indices = None,
                animals = animals,
                body = None,
                eyes = None,
                tail =  None,
                image = image
            )
            return res
        
        # loop over detected animals to get body, eyes and tail tracking
        body = {}
        eyes = {}
        tail = {}

        for id in animals.identities:

            bb_x, bb_y = animals.bb_centroids[id,:]
            left, bottom, right, top = animals.bounding_boxes[id,:]
            pad_left, pad_bottom, pad_right, pad_top = animals.padding[id,:]
            
            eyes[id] = None
            tail[id] = None
            body[id] = None

            # crop each animal's bounding box
            image_cropped = image[bottom:top, left:right] 

            # pad if image was clipped on the edges
            image_cropped = np.pad(image_cropped,((pad_bottom, pad_top),(pad_left, pad_right)))
            
            # bottom-left coordinate 
            offset = np.array([pad_left+bb_x, pad_bottom+bb_y])

            if self.body is not None:

                # get more precise centroid and orientation of the animals
                body[id] = self.body.track(image_cropped, centroid=offset)

                # if body was found, track eyes and tail
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

        # save tracking results and return
        res = MultiFishTracking(
            animals = animals,
            body = body,
            eyes = eyes,
            tail =  tail,
            image = image 
        )

        # compute additional features based on tracking
        if self.accumulator is not None:
            self.accumulator.update(res)

        return res 
    