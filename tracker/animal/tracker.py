from image_tools import bwareafilter_centroids_cv2
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalTracker
from tracker.prepare_image import preprocess_image
from geometry import SimilarityTransform2D

class AnimalTracker_CPU(AnimalTracker):
    
    def track(
        self,
        image: Optional[NDArray], # image in input space
        centroid: Optional[NDArray] = None, # centroid in global space
        T_input_to_global: Optional[SimilarityTransform2D] = SimilarityTransform2D.identity() # input to global space transform
    ) -> NDArray:
        

        if (image is None) or (image.size == 0):
            return self.tracking_param.failed
        
        preproc = preprocess_image(image, centroid, self.tracking_param)
        
        if preproc is None:
            return self.tracking_param.failed
        
        mask = cv2.compare(preproc.image_processed, self.tracking_param.intensity, cv2.CMP_GT)
        centroids_resized = bwareafilter_centroids_cv2(
            mask, 
            min_size = self.tracking_param.min_size_px,
            max_size = self.tracking_param.max_size_px, 
            min_length = self.tracking_param.min_length_px,
            max_length = self.tracking_param.max_length_px,
            min_width = self.tracking_param.min_width_px,
            max_width = self.tracking_param.max_width_px
        )     
        
        if centroids_resized.size == 0:
            return self.tracking_param.failed

        # transform coordinates
        centroids_cropped = preproc.T_resized_to_cropped.transform_points(centroids_resized)
        centroids_input = preproc.T_cropped_to_input.transform_points(centroids_cropped)
        centroids_global = T_input_to_global.transform_points(centroids_input)

        # identity assignment in global space
        centroids_global = self.assignment.update(centroids_global)  

        T_global_to_input = T_input_to_global.inv()

        centroids_input = T_global_to_input.transform_points(centroids_global)
        centroids_cropped = preproc.T_input_to_cropped.transform_points(centroids_input)
        centroids_resized = preproc.T_cropped_to_resized.transform_points(centroids_cropped)

        # Downsample image export (a bit easier on RAM). This is used for overlay instead of image_cropped
        # NOTE: it introduces a special case, not a big fan of this
        image_downsampled = cv2.resize(
            preproc.image_cropped,
            self.tracking_param.downsampled_shape[::-1], # transform shape (row, col) to width, height
            cv2.INTER_NEAREST
        )

        pix_per_mm_global = self.tracking_param.pix_per_mm
        pix_per_mm_input = pix_per_mm_global * T_global_to_input.scale_factor
        pix_per_mm_cropped = pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        pix_per_mm_resized = pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor
        pix_per_mm_downsampled = pix_per_mm_input * self.tracking_param.downsample_factor

        # NOTE: for large image/many fish, creating that array might take time
        res = np.array(
            (
                self.tracking_param.num_animals,
                centroids_resized,
                centroids_cropped,
                centroids_input,
                centroids_global, 
                self.tracking_param.downsample_factor,
                mask, 
                preproc.image_processed,
                image_downsampled,
                pix_per_mm_global,
                pix_per_mm_input,
                pix_per_mm_cropped,
                pix_per_mm_resized,
                pix_per_mm_downsampled
            ),
            dtype=self.tracking_param.dtype
        )
        return res
