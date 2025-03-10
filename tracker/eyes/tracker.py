import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import EyesTracker, DTYPE_EYE
from .utils import get_eye_properties, find_eyes_and_swimbladder, assign_features
from geometry import transform2d, Affine2DTransform, angle_between_vectors
from tracker.prepare_image import preprocess_image

class EyesTracker_CPU(EyesTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray], 
            transformation_matrix: Optional[NDArray] = Affine2DTransform.identity()
        ) -> NDArray:
        """
        output coordinates: 
            - (0,0) = fish centroid
            - scale of the full-resolution image, before resizing
        """

        failed = np.zeros((), dtype=self.tracking_param.dtype)

        if (image is None) or (image.size == 0) or (centroid is None):
            return failed
        
        preproc = preprocess_image(image, centroid, self.tracking_param)
        
        if preproc is None:
            return failed
        
        # sweep threshold to obtain 3 connected component within size range (include swim bladder)
        found_eyes_and_sb, props, mask = find_eyes_and_swimbladder(
            preproc.image_processed, 
            self.tracking_param.dyntresh_res, 
            self.tracking_param.size_lo_px, 
            self.tracking_param.size_hi_px,
            self.tracking_param.thresh_lo,
            self.tracking_param.thresh_hi
        )

        if not found_eyes_and_sb:
            return failed 
        
        # identify left eye, right eye and swimbladder
        blob_centroids = np.array([blob.centroid[::-1] for blob in props])
        swimbladder_idx, left_idx, right_idx = assign_features(blob_centroids)

        vertical_axis = np.array([0, 1], dtype=np.single)
        
        left_eye = get_eye_properties(
            props[left_idx], 
            preproc, 
            transformation_matrix, 
            vertical_axis
        )

        right_eye = get_eye_properties(
            props[right_idx], 
            preproc, 
            transformation_matrix, 
            vertical_axis
        )

        res = np.array(
            (
                np.zeros(1,dtype=DTYPE_EYE) if left_eye is None else left_eye, 
                np.zeros(1,dtype=DTYPE_EYE) if right_eye is None else right_eye,                
                mask, 
                preproc.image_processed,
                preproc.image_crop 
            ), 
            dtype = self.tracking_param.dtype
        )

        return res
    