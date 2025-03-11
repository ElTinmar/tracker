from image_tools import bwareafilter_centroids_cv2
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalTracker
from tracker.prepare_image import preprocess_image
from geometry import transform_point_2d, Affine2DTransform

class AnimalTracker_CPU(AnimalTracker):
    
    def track(
        self,
        image: Optional[NDArray], 
        centroid: Optional[NDArray] = None, # centroid in global space
        transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
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
        centroids_cropped = transform_point_2d(preproc.resize_transform, centroids_resized)
        centroids_input = transform_point_2d(preproc.crop_transform, centroids_cropped)
        centroids_global = transform_point_2d(transformation_matrix, centroids_input)

        # identity assignment in global space
        centroids_global = self.assignment.update(centroids_global)  

        centroids_input = transform_point_2d(np.linalg.inv(transformation_matrix), centroids_global)
        centroids_cropped = transform_point_2d(np.linalg.inv(preproc.crop_transform), centroids_input)
        centroids_resized = transform_point_2d(np.linalg.inv(preproc.resize_transform), centroids_cropped)

        # Downsample image export (a bit easier on RAM). This is used for overlay instead of image_cropped
        image_downsampled = cv2.resize(
            preproc.image_cropped,
            self.tracking_param.downsampled_shape[::-1], # transform shape (row, col) to width, height
            cv2.INTER_NEAREST
        )

        res = np.array(
            (
                self.tracking_param.num_animals,
                centroids_resized,
                centroids_cropped,
                centroids_input,
                centroids_global, 
                self.tracking_param.downsample_fullres,
                mask, 
                preproc.image_processed,
                image_downsampled
            ),
            dtype=self.tracking_param.dtype
        )
        return res
