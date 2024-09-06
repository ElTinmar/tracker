from image_tools import bwareafilter_centroids, bwareafilter_centroids_cv2, enhance, im2uint8
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalTracker

class AnimalTracker_CPU(AnimalTracker):
    
    def track(self, image: NDArray, centroid: Optional[NDArray] = None) -> Optional[NDArray]:

        if (image is None) or (image.size == 0):
            return None
        
        if self.tracking_param.resize != 1:
            image = cv2.resize(
                image, 
                self.tracking_param.image_shape,
                cv2.INTER_NEAREST
            )
        
        # tune image contrast and gamma
        image = enhance(
            image,
            self.tracking_param.animal_contrast,
            self.tracking_param.animal_gamma,
            self.tracking_param.animal_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        TRY_NEW_BWAREA = False
        if TRY_NEW_BWAREA:
            mask = cv2.compare(image, self.tracking_param.animal_intensity, cv2.CMP_GT)
            bwfun = bwareafilter_centroids_cv2
        else:
            mask = (image >= self.tracking_param.animal_intensity)
            bwfun = bwareafilter_centroids

        centroids = bwfun(
                mask, 
                min_size = self.tracking_param.min_animal_size_px,
                max_size = self.tracking_param.max_animal_size_px, 
                min_length = self.tracking_param.min_animal_length_px,
                max_length = self.tracking_param.max_animal_length_px,
                min_width = self.tracking_param.min_animal_width_px,
                max_width = self.tracking_param.max_animal_width_px
        )        

        if centroids.size != 0:
            # identity assignment
            self.assignment.update(centroids)
            identities = self.assignment.get_ID()
            indices_tokeep = self.assignment.get_kept_centroids()   
            centroids = centroids[indices_tokeep,:]/self.tracking_param.resize
        else:
            centroids = np.zeros((self.tracking_param.max_num_animals, 2), np.float32)
            identities = np.zeros((self.tracking_param.max_num_animals, 1), int)
            indices_tokeep = np.zeros((self.tracking_param.max_num_animals, 1), int)

        res = np.array(
            (
                identities is None,
                self.tracking_param.max_num_animals,
                identities,
                indices_tokeep, 
                centroids, 
                mask, 
                image
            ), 
            dtype=self.tracking_param.dtype()
        )
    
        return res
