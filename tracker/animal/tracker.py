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
        T_input_to_global: Optional[NDArray] = SimilarityTransform2D.identity() # input to global space transform
    ) -> NDArray:
        
        failed = np.zeros((), dtype=self.tracking_param.dtype)

        if (image is None) or (image.size == 0):
            return failed
        
        preproc = preprocess_image(image, centroid, self.tracking_param)
        
        if preproc is None:
            return failed
        
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
            return failed

        # transform coordinates
        centroids_cropped = transform_point_2d(preproc.T_resized_to_crop, centroids_resized)
        centroids_input = transform_point_2d(preproc.T_cropped_to_input, centroids_cropped)
        centroids_global = transform_point_2d(T_input_to_global, centroids_input)

        # identity assignment in global space
        centroids_global = self.assignment.update(centroids_global)  

        T_global_to_input = np.linalg.inv(T_input_to_global)
        T_input_to_cropped = np.linalg.inv(preproc.T_cropped_to_input)
        T_cropped_to_resized = np.linalg.inv(preproc.T_resized_to_crop)

        centroids_input = transform_point_2d(T_global_to_input, centroids_global)
        centroids_cropped = transform_point_2d(T_input_to_cropped, centroids_input)
        centroids_resized = transform_point_2d(T_cropped_to_resized, centroids_cropped)

        # Downsample image export (a bit easier on RAM). This is used for overlay instead of image_cropped
        # NOTE: it introduces a special case, not a big fan of this
        image_downsampled = cv2.resize(
            preproc.image_cropped,
            self.tracking_param.downsampled_shape[::-1], # transform shape (row, col) to width, height
            cv2.INTER_NEAREST
        )

        # This works if isotropy is preserved (same x,y scale)
        # T_input_to_global could break that
        pix_per_mm_global = self.tracking_param.pix_per_mm
        pix_per_mm_input = pix_per_mm_global * np.linalg.norm(T_global_to_input[:,0])
        pix_per_mm_cropped = pix_per_mm_input * np.linalg.norm(T_input_to_cropped[:,0]) 
        pix_per_mm_resized = pix_per_mm_cropped * np.linalg.norm(T_cropped_to_resized[:,0]) 
        pix_per_mm_downsampled = pix_per_mm_input * self.tracking_param.downsample_factor

        if not np.isclose(pix_per_mm_resized, self.tracking_param.target_pix_per_mm):
            print(f'scaling problem, {pix_per_mm_resized} vs {self.tracking_param.target_pix_per_mm}')

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
