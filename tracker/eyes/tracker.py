import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import EyesTracker, DTYPE_EYE
from .utils import get_eye_prop, get_eye_prop_cv2, find_eyes_and_swimbladder, assign_features
from tracker.prepare_image import prepare_image
from geometry import Affine2DTransform

TRY_NEW_BWAREA = True

class EyesTracker_CPU(EyesTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray], 
            transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
        ) -> Optional[NDArray]:
        """
        output coordinates: 
            - (0,0) = fish centroid
            - scale of the full-resolution image, before resizing
        """

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
            blob_centroids = np.array([blob.centroid[::-1] for blob in props])
            sb_idx, left_idx, right_idx = assign_features(blob_centroids)
            centroid_left = np.asarray(props[left_idx].centroid[::-1], dtype=np.float32)
            centroid_right = np.asarray(props[right_idx].centroid[::-1], dtype=np.float32)
            centroid_sb = np.asarray(props[sb_idx].centroid[::-1], dtype=np.float32)
                
            # compute eye orientation
            if TRY_NEW_BWAREA:
                left_eye = get_eye_prop_cv2(
                    centroid_left, 
                    props[left_idx].principal_axis, 
                    origin*self.tracking_param.resize,
                    self.tracking_param.resize,
                    transformation_matrix
                )
                right_eye = get_eye_prop_cv2(
                    centroid_right, 
                    props[right_idx].principal_axis,
                    origin*self.tracking_param.resize,
                    self.tracking_param.resize,
                    transformation_matrix
                )

            else:
                left_eye = get_eye_prop(
                    centroid_left, 
                    props[left_idx].inertia_tensor, 
                    origin*self.tracking_param.resize,
                    self.tracking_param.resize,
                    transformation_matrix
                )
                right_eye = get_eye_prop(
                    centroid_right, 
                    props[right_idx].inertia_tensor,
                    origin*self.tracking_param.resize,
                    self.tracking_param.resize,
                    transformation_matrix
                )

            heading_vector = (centroid_left + centroid_right)/2 - centroid_sb
            heading_vector = heading_vector / np.linalg.norm(heading_vector)
            #heading_vector_original_space = 

        res = np.array(
            (
                centroid is None,
                np.zeros((1,2), np.float32) if centroid is None else centroid,
                np.zeros((1,2), np.float32) if heading_vector is None else heading_vector,
                np.zeros((1,2), np.int32) if origin is None else origin,
                np.zeros(1,dtype=DTYPE_EYE) if left_eye is None else left_eye, 
                np.zeros(1,dtype=DTYPE_EYE) if right_eye is None else right_eye,                
                mask, 
                image_processed,
                image_crop 
            ), 
            dtype = self.tracking_param.dtype()
        )

        return res
    