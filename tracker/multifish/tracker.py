import numpy as np
from typing import Optional
from numpy.typing import NDArray
from image_tools import imrotate
from .core import MultiFishTracker
from geometry import SimilarityTransform2D

class MultiFishTracker_CPU(MultiFishTracker):

    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[NDArray]:

        if (image is None) or (image.size == 0):
            return None

        # get animal centroids (only crude location is necessary)
        animals = self.tracking_param.animal.track(image, None, SimilarityTransform2D.identity())
        
        bodies = []
        eyes = []
        tails = []

        # loop over detected animals to get body, eyes and tail tracking
        for centroid in animals['centroids_global']:

            if self.tracking_param.body is not None:

                # get more precise centroid and orientation of the animals
                body = self.tracking_param.body.track(image, centroid, SimilarityTransform2D.identity())
                bodies.append(body)
                    
                # rotate the animal so that it's vertical head up
                image_rot, centroid_rot = imrotate(
                    body['image_cropped'], 
                    body['centroid_cropped'][0], body['centroid_cropped'][1], 
                    np.rad2deg(body['angle_rad'])
                )

                R = SimilarityTransform2D.rotation(body['angle_rad'])
                T = SimilarityTransform2D.translation(body['centroid_global'][0], body['centroid_global'][1])
                T0 = SimilarityTransform2D.translation(-centroid_rot[0], -centroid_rot[1])
                T_image_rot_to_global =  T @ R @ T0
            
                # track eyes
                if self.tracking_param.eyes is not None:
                    eyes.append(self.tracking_param.eyes.track(image_rot, centroid_rot, T_image_rot_to_global))

                # track tail
                if self.tracking_param.tail is not None:
                    tails.append(self.tracking_param.tail.track(image_rot, centroid_rot, T_image_rot_to_global))

        # compute additional features based on tracking
        if self.tracking_param.accumulator is not None:
            self.tracking_param.accumulator.update(res)

        # save tracking results and return
        arr = (animals,)

        if self.tracking_param.body is not None:
            arr += (bodies,)

        if self.tracking_param.eyes is not None:
            arr += (eyes,)

        if self.tracking_param.tail is not None:
            arr += (tails,)

        try:
            res = np.array(
                arr,
                dtype=self.tracking_param.dtype
            )
        except ValueError:
            # FIXME shape (0,) cannot be broadcast to (1,)
            # this may happen if you try to get eyes or tail without body
            print(len(bodies), len(eyes), len(tails))
            raise
            return None

        return res 
    