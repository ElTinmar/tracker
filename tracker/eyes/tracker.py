import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import EyesTracker, EyesTracking
from .utils import get_eye_prop, find_eyes_and_swimbladder, assign_features
from tracker.prepare_image import prepare_image

class EyesTracker_CPU(EyesTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray],
        ) -> Optional[EyesTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
            return None
        
        # pre-process image: crop/resize/tune intensity
        (origin, image_crop, image_processed) = prepare_image(
            image=image,
            source_crop_dimension_px=self.tracking_param.source_crop_dimension_px,
            target_crop_dimension_px=self.tracking_param.crop_dimension_px, 
            vertical_offset_px=self.tracking_param.crop_offset_px, 
            centroid=centroid,
            contrast=self.tracking_param.eye_contrast,
            gamma=self.tracking_param.eye_gamma,
            brightness=self.tracking_param.eye_brightness,
            blur_sz_px=self.tracking_param.blur_sz_px,
            median_filter_sz_px=self.tracking_param.median_filter_sz_px
        )

        # sweep threshold to obtain 3 connected component within size range (include swim bladder)
        found_eyes_and_sb, props, mask = find_eyes_and_swimbladder(
            image_processed, 
            self.tracking_param.eye_dyntresh_res, 
            self.tracking_param.eye_size_lo_px, 
            self.tracking_param.eye_size_hi_px,
            self.tracking_param.eye_thresh_lo,
            self.tracking_param.eye_thresh_hi
        )

        # find eye angles
        left_eye = None
        right_eye = None
        heading_vector = None
        
        if found_eyes_and_sb: 
            # identify left eye, right eye and swimbladder
            blob_centroids = np.array([blob.centroid for blob in props])
            sb_idx, left_idx, right_idx = assign_features(blob_centroids)

            # compute eye orientation
            left_eye = get_eye_prop(
                props[left_idx].centroid, 
                props[left_idx].inertia_tensor, 
                origin*self.tracking_param.resize,
                self.tracking_param.resize
            )
            right_eye = get_eye_prop(
                props[right_idx].centroid, 
                props[right_idx].inertia_tensor,
                origin*self.tracking_param.resize,
                self.tracking_param.resize
            )
            heading_vector = (props[left_idx].centroid + props[right_idx].centroid)/2 - props[sb_idx].centroid
            heading_vector = heading_vector / np.linalg.norm(heading_vector)

        res = EyesTracking(
            im_eyes_shape = image_processed.shape,
            im_eyes_fullres_shape = image_crop.shape,
            centroid = centroid,
            heading_vector = heading_vector,
            origin = origin,
            left_eye = left_eye,
            right_eye = right_eye,
            mask = mask,
            image_fullres = image_crop,
            image = image_processed
        )

        return res
    