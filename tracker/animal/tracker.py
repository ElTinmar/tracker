from image_tools import bwareafilter_centroids_cv2
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalTracker
from tracker.prepare_image import preprocess_image
from geometry import Affine2DTransform

class AnimalTracker_CPU(AnimalTracker):
    
    def track(
        self,
        image: Optional[NDArray], 
        centroid: Optional[NDArray] = None,
        transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
    ) -> Optional[NDArray]:

        if (image is None) or (image.size == 0):
            return None
        
        image_crop, image_resized, image_processed = preprocess_image(image, centroid, self.tracking_param)

        mask = cv2.compare(image_processed, self.tracking_param.intensity, cv2.CMP_GT)
        centroids_resized = bwareafilter_centroids_cv2(
            mask, 
            min_size = self.tracking_param.min_animal_size_px,
            max_size = self.tracking_param.max_animal_size_px, 
            min_length = self.tracking_param.min_animal_length_px,
            max_length = self.tracking_param.max_animal_length_px,
            min_width = self.tracking_param.min_animal_width_px,
            max_width = self.tracking_param.max_animal_width_px
        )     
        
        if centroids_resized.size == 0:
            return None
        
        centroids_cropped = centroids_resized/self.tracking_param.resize 

        # identity assignment
        self.assignment.update(centroids)
        identities = self.assignment.get_ID()
        indices_tokeep = self.assignment.get_kept_centroids()   
        centroids = self.assignment.get_centroids()

        # Downsample image export. This is a bit easier on RAM
        image_export = cv2.resize(
            image,
            self.tracking_param.downsampled_shape[::-1], # transform shape (row, col) to width, height
            cv2.INTER_NEAREST
        )

        res = np.array(
            (
                identities is None,
                self.tracking_param.num_animals,
                identities,
                indices_tokeep, 
                centroids, 
                mask, 
                image_processed,
                image_export,
                self.tracking_param.downsample_fullres
            ), 
            dtype=self.tracking_param.dtype()
        )
        return res
