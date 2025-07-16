import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
from image_tools import imrotate, im2gray
from .core import SingleFishTracker
from geometry import SimilarityTransform2D

class SingleFishTracker_CPU(SingleFishTracker):

    def track(
            self, 
            image: NDArray, 
            background_image: Optional[NDArray] = None,
            centroid: Optional[NDArray] = None,
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:
        
        # only work with one channel
        image = im2gray(image)

        if background_image is None:
            background_image = np.zeros_like(image)
        background_image = im2gray(background_image)
        
        # get animal centroids (only crude location is necessary)
        animals = self.tracking_param.animal.track(
            image, 
            background_image, 
            None, 
            T_input_to_global
        )

        if not animals['success']:
            return self.tracking_param.failed
        
        arr = (True, animals)
        centroid = animals['centroids_global'][0,:]
        
        body = eyes = tail = None
        if self.tracking_param.body is not None:

            # get more precise centroid and orientation of the animals
            body = self.tracking_param.body.track(
                image, 
                background_image, 
                centroid, 
                T_input_to_global
            )
            arr += (body,)

            # rotate the animal so that it's vertical head up
            image_rot, centroid_rot = imrotate(
                body['image_cropped'], 
                body['centroid_cropped'][0], body['centroid_cropped'][1], 
                float(np.rad2deg(body['angle_rad']))  
            )
            background_rot, _ = imrotate(
                body['background_image_cropped'], 
                body['centroid_cropped'][0], body['centroid_cropped'][1], 
                float(np.rad2deg(body['angle_rad'])) 
            )

            T = SimilarityTransform2D.translation(body['centroid_input'][0], body['centroid_input'][1])
            R = SimilarityTransform2D.rotation(float(body['angle_rad']))
            T0 = SimilarityTransform2D.translation(-centroid_rot[0], -centroid_rot[1])
            
            T_image_rot_to_global =  T_input_to_global @ T @ R @ T0
        
            # track eyes
            if self.tracking_param.eyes is not None:
                eyes = self.tracking_param.eyes.track(
                    image_rot, 
                    background_rot, 
                    centroid, 
                    T_image_rot_to_global
                )
                arr += (eyes,)

            # track tail
            if self.tracking_param.tail is not None:
                tail = self.tracking_param.tail.track(
                    image_rot, 
                    background_rot, 
                    centroid, 
                    T_image_rot_to_global
                )
                arr += (tail,)

        res = np.array(
            arr,
            dtype=self.tracking_param.dtype
        )

        return res
    