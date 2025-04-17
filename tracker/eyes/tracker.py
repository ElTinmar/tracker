import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from .core import EyesTracker, DTYPE_EYE
from .utils import get_eye_properties, find_eyes_and_swimbladder, assign_features
from geometry import SimilarityTransform2D
from tracker.prepare_image import preprocess_image
from filterpy.common import kinematic_kf

class EyesTracker_CPU(EyesTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray], 
            T_input_to_global: Optional[SimilarityTransform2D] = SimilarityTransform2D.identity()
        ) -> NDArray:
        """
        output coordinates: 
            - (0,0) = fish centroid
            - scale of the full-resolution image, before resizing
        """

        self.tracking_param.input_image_shape = image.shape

        if centroid is None:
            return self.tracking_param.failed
        
        T_global_to_input = T_input_to_global.inv()
        centroid_input = T_global_to_input.transform_points(centroid).squeeze()
        preproc = preprocess_image(image, centroid_input, self.tracking_param)
        
        if preproc is None:
            return self.tracking_param.failed
        
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
            return self.tracking_param.failed
        
        # identify left eye, right eye and swimbladder
        blob_centroids = np.array([blob.centroid[::-1] for blob in props])
        swimbladder_idx, left_idx, right_idx = assign_features(blob_centroids)

        vertical_axis = np.array([0, 1], dtype=np.single)
        
        left_eye = get_eye_properties(
            props[left_idx], 
            preproc, 
            T_input_to_global, 
            vertical_axis
        )

        right_eye = get_eye_properties(
            props[right_idx], 
            preproc, 
            T_input_to_global, 
            vertical_axis
        )

        pix_per_mm_global = self.tracking_param.pix_per_mm
        pix_per_mm_input = pix_per_mm_global * T_global_to_input.scale_factor
        pix_per_mm_cropped = pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        pix_per_mm_resized = pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor

        res = np.array(
            (
                True,
                np.zeros(1,dtype=DTYPE_EYE) if left_eye is None else left_eye, 
                np.zeros(1,dtype=DTYPE_EYE) if right_eye is None else right_eye,                
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                pix_per_mm_global,
                pix_per_mm_input,
                pix_per_mm_cropped,
                pix_per_mm_resized,
            ), 
            dtype = self.tracking_param.dtype
        )

        return res
    
class EyesTrackerKalman(EyesTracker_CPU):

    N_DIM = 6

    def __init__(
            self, 
            fps: int, 
            model_order: int, 
            model_uncertainty: float = 0.2,
            measurement_uncertainty: float = 1.0,
            *args, 
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.fps = fps
        dt = 1/fps
        self.kalman_filter = kinematic_kf(
            dim = self.N_DIM, 
            order = model_order, 
            dt = dt, 
            dim_z = self.N_DIM, 
            order_by_dim = False
        )
        self.kalman_filter.Q *= model_uncertainty
        self.kalman_filter.R *= measurement_uncertainty

    def tracking_to_measurement(self, tracking: NDArray) -> NDArray:
        
        if tracking['success']:
            measurement = np.zeros((self.N_DIM,1))
            measurement[:2,0] = tracking['left_eye']['centroid_resized']
            measurement[2] = tracking['left_eye']['angle']
            measurement[3:5,0] = tracking['right_eye']['centroid_resized']
            measurement[5] = tracking['right_eye']['angle']
        else:
            measurement = None

        return measurement

    def prediction_to_tracking(self, tracking: NDArray) -> None:
        '''Side effect: modify tracking in-place'''
    
        tracking['left_eye']['centroid_resized'] = self.kalman_filter.x[:2,0]
        tracking['left_eye']['angle'] = self.kalman_filter.x[2]
        tracking['left_eye']['direction'] = np.array([
            np.sin(tracking['left_eye']['angle']), 
            np.cos(tracking['left_eye']['angle'])
        ])

        tracking['right_eye']['centroid_resized'] = self.kalman_filter.x[3:5,0]
        tracking['right_eye']['angle'] = self.kalman_filter.x[5]
        tracking['right_eye']['direction'] = np.array([
            np.sin(tracking['right_eye']['angle']), 
            np.cos(tracking['right_eye']['angle'])
        ])

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: Optional[SimilarityTransform2D] = SimilarityTransform2D.identity()
        ) -> NDArray:

        tracking = super().track(image, centroid, T_input_to_global)
        self.kalman_filter.predict()
        measurement = self.tracking_to_measurement(tracking)
        self.kalman_filter.update(measurement)
        self.prediction_to_tracking(tracking)
        
        return tracking
    