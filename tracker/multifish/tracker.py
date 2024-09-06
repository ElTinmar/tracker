import numpy as np
from typing import Optional
from numpy.typing import NDArray
from image_tools import imrotate
from .core import MultiFishTracker, MultiFishTracking
import cv2

class MultiFishTracker_CPU(MultiFishTracker):

    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[MultiFishTracking]:

        if (image is None) or (image.size == 0):
            return None

        # get animal centroids (only crude location is necessary)
        animals = self.tracking_param.animal.track(image)

        body = []
        eyes = []
        tail = []

        # loop over detected animals to get body, eyes and tail tracking

        if animals.identities is not None:
            for id in animals.identities:
                if self.tracking_param.body is not None:

                    # get more precise centroid and orientation of the animals
                    b = self.tracking_param.body.track(image, centroid=animals.centroids[id,:])
                    body.append(b)

                    # if body was found, track eyes and tail
                    if (b is not None) and (b['centroid'] is not None):
                        
                        # rotate the animal so that it's vertical head up
                        image_rot, centroid_rot = imrotate(
                            b['image_fullres'], 
                            b['centroid'][0], b['centroid'][1], 
                            np.rad2deg(b['angle_rad'])
                        )

                        # track eyes
                        if self.tracking_param.eyes is not None:
                            eyes.append(self.tracking_param.eyes.track(image_rot, centroid=centroid_rot))

                        # track tail
                        if self.tracking_param.tail is not None:
                            tail.append(self.tracking_param.tail.track(image_rot, centroid=centroid_rot))

        # compute additional features based on tracking
        if self.tracking_param.accumulator is not None:
            self.tracking_param.accumulator.update(res)

        # save tracking results and return
        arr = [animals]
        
        if self.tracking_param.body is not None:
            arr += [body]

        if self.tracking_param.eyes is not None:
            arr += [eyes]

        if self.tracking_param.tail is not None:
            arr += [tail]

        res = np.array(
            arr,
            dtype=self.tracking_param.dtype()
        )

        return res 
    