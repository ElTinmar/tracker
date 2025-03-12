from image_tools import  bwareafilter_props_cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import BodyTracker
from .utils import get_orientation, get_best_centroid_index
from tracker.prepare_image import preprocess_image
from geometry import SimilarityTransform2D
import cv2

class BodyTracker_CPU(BodyTracker):
        
    def track(
            self,
            image: Optional[NDArray], 
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: Optional[SimilarityTransform2D] = SimilarityTransform2D.identity()
        ) -> NDArray:
        '''
        centroid: centroid of the fish to track if it's already known.
        Useful when tracking multiple fish to discriminate between nearby blobs

        output coordinates: 
            - (0,0) = topleft corner of the bounding box
            - scale of the full-resolution image, before resizing
        '''

        failed = np.zeros((), dtype=self.tracking_param.dtype)

        if (image is None) or (image.size == 0):
            return failed
        
        preproc = preprocess_image(image, centroid, self.tracking_param)
        
        if preproc is None:
            return failed

        mask = cv2.compare(preproc.image_processed, self.tracking_param.intensity, cv2.CMP_GT)
        props = bwareafilter_props_cv2(
            mask, 
            min_size = self.tracking_param.min_size_px,
            max_size = self.tracking_param.max_size_px, 
            min_length = self.tracking_param.min_length_px,
            max_length = self.tracking_param.max_length_px,
            min_width = self.tracking_param.min_width_px,
            max_width = self.tracking_param.max_width_px
        )

        if not props:
            return failed
        
        centroids_resized = np.array([[blob.centroid[1], blob.centroid[0]] for blob in props]) #(row, col) to (x,y)
        centroids_cropped = preproc.T_resized_to_crop.transform_points(centroids_resized)
        centroids_input = preproc.T_cropped_to_input.transform_points(centroids_cropped)
        centroids_global = T_input_to_global.transform_points(centroids_input)

        # get coordinates of best centroid
        index = get_best_centroid_index(centroids_global, centroid)
        centroid_resized = centroids_resized[index]
        centroid_cropped = centroids_cropped[index]
        centroid_input = centroids_input[index]
        centroid_global = centroids_global[index]

        coordinates_resized = np.fliplr(props[index].coords)
        body_axes = get_orientation(coordinates_resized)
        if body_axes is None:
            return failed
        
        body_axes_global = T_input_to_global.transform_vectors(body_axes) 
        
        angle_rad = np.arctan2(body_axes[1,1], body_axes[0,1])
        angle_rad_global = np.arctan2(body_axes_global[1,1], body_axes_global[0,1])

        T_global_to_input = T_input_to_global.inv()
        T_input_to_cropped = preproc.T_cropped_to_input.inv()
        T_cropped_to_resized = preproc.T_resized_to_crop.inv()
        pix_per_mm_global = self.tracking_param.pix_per_mm
        pix_per_mm_input = pix_per_mm_global * T_global_to_input.scale_factor
        pix_per_mm_cropped = pix_per_mm_input * T_input_to_cropped.scale_factor
        pix_per_mm_resized = pix_per_mm_cropped * T_cropped_to_resized.scale_factor
        
        res = np.array(
            (
                body_axes, 
                body_axes_global,
                centroid_resized,
                centroid_cropped,
                centroid_input,
                centroid_global,
                angle_rad,
                angle_rad_global,
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                pix_per_mm_global,
                pix_per_mm_input,
                pix_per_mm_cropped,
                pix_per_mm_resized,
            ), 
            dtype=self.tracking_param.dtype
        )
        return res
