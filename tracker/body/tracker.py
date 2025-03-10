from image_tools import  bwareafilter_props_cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import BodyTracker
from .utils import get_orientation, get_best_centroid_index
from tracker.prepare_image import preprocess_image
from geometry import transform2d, Affine2DTransform
import cv2

class BodyTracker_CPU(BodyTracker):
        
    def track(
            self,
            image: Optional[NDArray], 
            centroid: Optional[NDArray] = None, # centroids in global space
            transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
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
        centroids_cropped = transform2d(preproc.resize_transform, centroids_resized)
        centroids_input = transform2d(preproc.crop_transform, centroids_cropped)
        centroids_global = transform2d(transformation_matrix, centroids_input)

        # get coordinates of best centroid
        index = get_best_centroid_index(centroids_global, centroid)
        centroid_resized = centroids_resized[index]
        centroid_cropped = centroids_cropped[index]
        centroid_input = centroids_input[index]
        centroid_global = centroids_global[index]

        coordinates_resized = props[index].coords[::-1]
        principal_components = get_orientation(coordinates_resized)
        if principal_components is None:
            return failed
        
        principal_components_global = transform2d(transformation_matrix, principal_components)
        
        angle_rad = np.arctan2(principal_components[1,1], principal_components[0,1])
        angle_rad_global = np.arctan2(principal_components_global[1,1], principal_components_global[0,1])
        
        res = np.array(
            (
                principal_components, 
                principal_components_global,
                centroid_resized,
                centroid_cropped,
                centroid_input,
                centroid_global,
                angle_rad,
                angle_rad_global,
                mask, 
                preproc.image_processed,
                preproc.image_crop
            ), 
            dtype=self.tracking_param.dtype
        )
        return res
