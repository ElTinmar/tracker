import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import enhance, im2uint8
from .core import EyesTracker, EyesTracking
from .utils import get_eye_prop, find_eyes_and_swimbladder, assign_features

class EyesTracker_CPU(EyesTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray],
        ) -> Optional[EyesTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
            return None

        if self.tracking_param.resize != 1:
            image = cv2.resize(
                image, 
                None, 
                None,
                self.tracking_param.resize,
                self.tracking_param.resize,
                cv2.INTER_NEAREST
            )

        left_eye = None
        right_eye = None
        new_heading = None

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        offset = np.array((-w//2, -h//2+self.tracking_param.crop_offset_px), dtype=np.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(np.int32) + offset 
        right, top = left+w, bottom+h 

        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance(
            image_crop,
            self.tracking_param.eye_contrast,
            self.tracking_param.eye_gamma,
            self.tracking_param.eye_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # sweep threshold to obtain 3 connected component within size range (include swim bladder)
        found_eyes_and_sb, props, mask = find_eyes_and_swimbladder(
            image_crop, 
            self.tracking_param.eye_dyntresh_res, 
            self.tracking_param.eye_size_lo_px, 
            self.tracking_param.eye_size_hi_px
        )
        
        if found_eyes_and_sb: 
            # identify left eye, right eye and swimbladder
            blob_centroids = np.array([blob.centroid for blob in props])
            sb_idx, left_idx, right_idx = assign_features(blob_centroids)

            # compute eye orientation
            left_eye = get_eye_prop(
                props[left_idx].centroid, 
                props[left_idx].inertia_tensor, 
                offset, 
                self.tracking_param.resize
            )
            right_eye = get_eye_prop(
                props[right_idx].centroid, 
                props[right_idx].inertia_tensor,
                offset, 
                self.tracking_param.resize
            )
            #new_heading = (props[left_idx].centroid + props[right_idx].centroid)/2 - props[sb_idx].centroid
            #new_heading = new_heading / np.linalg.norm(new_heading)

        res = EyesTracking(
            centroid = centroid,
            offset = offset,
            left_eye = left_eye,
            right_eye = right_eye,
            mask = im2uint8(mask),
            image = im2uint8(image_crop)
        )

        return res
    