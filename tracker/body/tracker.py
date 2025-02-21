from image_tools import  bwareafilter_props, bwareafilter_props_cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import BodyTracker
from .utils import get_orientation, get_blob_coordinates
from tracker.prepare_image import prepare_image
from geometry import transform2d, Affine2DTransform
import cv2

class BodyTracker_CPU(BodyTracker):
        
    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] = None,
            transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
        ) -> NDArray:
        '''
        centroid: centroid of the fish to track if it's already known.
        Useful when tracking multiple fish to discriminate between nearby blobs

        output coordinates: 
            - (0,0) = topleft corner of the bounding box
            - scale of the full-resolution image, before resizing
        '''

        if (image is None) or (image.size == 0):
            return None
        
        # pre-process image: crop/resize/tune intensity
        (origin, image_crop, image_processed) = prepare_image(
            image=image,
            source_crop_dimension_px=self.tracking_param.source_crop_dimension_px,
            target_crop_dimension_px=self.tracking_param.crop_dimension_px, 
            vertical_offset_px=0,
            centroid=centroid,
            contrast=self.tracking_param.body_contrast,
            gamma=self.tracking_param.body_gamma,
            brightness=self.tracking_param.body_brightness,
            blur_sz_px=self.tracking_param.blur_sz_px,
            median_filter_sz_px=self.tracking_param.median_filter_sz_px
        )
    
        # actual tracking starts here
        mask = cv2.compare(image_processed, self.tracking_param.body_intensity, cv2.CMP_GT)
        props = bwareafilter_props_cv2(
            mask, 
            min_size = self.tracking_param.min_body_size_px,
            max_size = self.tracking_param.max_body_size_px, 
            min_length = self.tracking_param.min_body_length_px,
            max_length = self.tracking_param.max_body_length_px,
            min_width = self.tracking_param.min_body_width_px,
            max_width = self.tracking_param.max_body_width_px
        )
        
        angle_rad = None
        principal_components = None
        centroid_coords = None
        centroid_ori = None

        if props != []:
            coordinates = get_blob_coordinates(centroid, props, self.tracking_param.resize)
            if coordinates.shape[0] > 1:
                (principal_components, centroid_coords) = get_orientation(coordinates)

                if principal_components is not None:
                    angle_rad = np.arctan2(principal_components[1,1], principal_components[0,1])
                
                if (centroid is not None) and (centroid_coords is not None):
                    centroid_ori = origin + centroid + centroid_coords / self.tracking_param.resize 

        res = np.array(
            (
                principal_components is None,
                np.zeros((2,2), np.float32) if principal_components is None else principal_components, 
                np.zeros((1,2), np.float32) if centroid_coords is None else centroid_coords / self.tracking_param.resize,
                np.zeros((1,2), np.float32) if centroid_ori is None else transform2d(transformation_matrix, centroid_ori),
                np.zeros((1,2), np.float32) if origin is None else origin,
                0.0 if angle_rad is None else angle_rad, 
                mask, 
                image_processed,
                image_crop
            ), 
            dtype=self.tracking_param.dtype()
        )
        return res
