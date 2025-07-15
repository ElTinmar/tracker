import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from .core import EyesTracker, DTYPE_EYE
from .utils import get_eye_properties, find_eyes_and_swimbladder, assign_features
from geometry import SimilarityTransform2D, angle_between_vectors
from tracker.prepare_image import preprocess_image, Preprocessing
from tracker.core import Resolution
from filterpy.common import kinematic_kf
from dataclasses import dataclass

@dataclass
class Tracking:
    left_eye: NDArray = np.zeros((), dtype=DTYPE_EYE)
    right_eye: NDArray = np.zeros((), dtype=DTYPE_EYE)

class EyesTracker_CPU(EyesTracker):

    vertical_axis = np.array([0, 1], dtype=np.single)

    def transform_input_centroid(
            self, 
            centroid, # centroid in global space
            T_input_to_global
        ) -> Tuple[Optional[NDArray], SimilarityTransform2D]:
        
        T_global_to_input = T_input_to_global.inv()

        if centroid is None:
            centroid_input = None
        else:
            centroid_input = T_global_to_input.transform_points(centroid).squeeze()

        return centroid_input, T_global_to_input      

    def preprocess(
        self,
        image: NDArray, 
        background_image: NDArray, 
        centroid: Optional[NDArray] = None, # centroids in input space
        ) -> Optional[Preprocessing]:
        
        return preprocess_image(image, background_image, centroid, self.tracking_param)


    def track_resized(
            self,
            preproc: Preprocessing, 
        ) -> Optional[Tuple[Tracking, NDArray]]:

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
            return None
        
        tracking = Tracking()

        # identify left eye, right eye and swimbladder
        blob_centroids = np.array([blob.centroid[::-1] for blob in props])
        swimbladder_idx, left_idx, right_idx = assign_features(blob_centroids)
        
        tracking.left_eye = get_eye_properties(
            props[left_idx], 
            self.vertical_axis
        )
        if tracking.left_eye is None:
            return None 

        tracking.right_eye = get_eye_properties(
            props[right_idx], 
            self.vertical_axis
        )
        if tracking.right_eye is None:
            return None 

        return tracking, mask

    def transform_coordinate_system(
            self,
            tracking: Tracking,        
            preproc: Preprocessing,
            T_input_to_global,
            T_global_to_input
        ) -> Resolution:
        '''modifies tracking in-place with side-effect and returns resolution'''
       
        tracking.left_eye['direction_global'] = T_input_to_global.transform_vectors(tracking.left_eye['direction']) 
        tracking.left_eye['angle_global'] = angle_between_vectors(tracking.left_eye['direction_global'], self.vertical_axis)
        tracking.left_eye['centroid_cropped'] = preproc.T_resized_to_cropped.transform_points(tracking.left_eye['centroid_resized'])
        tracking.left_eye['centroid_input'] =  preproc.T_cropped_to_input.transform_points(tracking.left_eye['centroid_cropped'])       
        tracking.left_eye['centroid_global'] = T_input_to_global.transform_points(tracking.left_eye['centroid_input'])

        tracking.right_eye['direction_global'] = T_input_to_global.transform_vectors(tracking.right_eye['direction']) 
        tracking.right_eye['angle_global'] = angle_between_vectors(tracking.right_eye['direction_global'], self.vertical_axis)
        tracking.right_eye['centroid_cropped'] = preproc.T_resized_to_cropped.transform_points(tracking.right_eye['centroid_resized'])
        tracking.right_eye['centroid_input'] =  preproc.T_cropped_to_input.transform_points(tracking.right_eye['centroid_cropped'])       
        tracking.right_eye['centroid_global'] = T_input_to_global.transform_points(tracking.right_eye['centroid_input'])

        resolution = Resolution()
        resolution.pix_per_mm_global = self.tracking_param.pix_per_mm
        resolution.pix_per_mm_input = resolution.pix_per_mm_global * T_global_to_input.scale_factor
        resolution.pix_per_mm_cropped = resolution.pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        resolution.pix_per_mm_resized = resolution.pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor

        return resolution
    
    def track(
            self,
            image: NDArray, 
            background_image: Optional[NDArray] = None, 
            centroid: Optional[NDArray] = None, 
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:
        """
        output coordinates: 
            - (0,0) = fish centroid
            - scale of the full-resolution image, before resizing
        """

        if background_image is None:
            background_image = np.zeros_like(image)

        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )

        preproc = self.preprocess(image, background_image, centroid_in_input) 
        if preproc is None:
            return self.tracking_param.failed
        
        tracking_resized = self.track_resized(preproc)
        if tracking_resized is None:
            return self.tracking_param.failed
        
        tracking, mask = tracking_resized
        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )

        res = np.array(
            (
                True,
                np.zeros(1,dtype=DTYPE_EYE) if tracking.left_eye is None else tracking.left_eye, 
                np.zeros(1,dtype=DTYPE_EYE) if tracking.right_eye is None else tracking.right_eye,                
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
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

    def tracking_to_measurement(self, tracking: Tracking) -> NDArray:
        
        measurement = np.zeros((self.N_DIM,1))
        measurement[:2,0] = tracking.left_eye['centroid_resized']
        measurement[2,0] = tracking.left_eye['angle']
        measurement[3:5,0] = tracking.right_eye['centroid_resized']
        measurement[5,0] = tracking.right_eye['angle']

        return measurement

    def prediction_to_tracking(self, tracking: Tracking) -> None:
        '''Side effect: modify tracking in-place'''
    
        tracking.left_eye['centroid_resized'] = self.kalman_filter.x[:2,0]
        tracking.left_eye['angle'] = self.kalman_filter.x[2,0]
        tracking.left_eye['direction'] = np.array([
            np.sin(tracking.left_eye['angle']), 
            np.cos(tracking.left_eye['angle'])
        ])

        tracking.right_eye['centroid_resized'] = self.kalman_filter.x[3:5,0]
        tracking.right_eye['angle'] = self.kalman_filter.x[5,0]
        tracking.right_eye['direction'] = np.array([
            np.sin(tracking.right_eye['angle']), 
            np.cos(tracking.right_eye['angle'])
        ])

    def return_prediction_if_tracking_failed(
            self,
            preproc: Preprocessing,
            T_input_to_global: SimilarityTransform2D,
            T_global_to_input: SimilarityTransform2D,
        ) -> NDArray:
        
        tracking = Tracking()
        self.kalman_filter.predict()
        self.kalman_filter.update(None)
        self.prediction_to_tracking(tracking)

        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )

        res = np.array(
            (
                True,
                np.zeros(1,dtype=DTYPE_EYE) if tracking.left_eye is None else tracking.left_eye, 
                np.zeros(1,dtype=DTYPE_EYE) if tracking.right_eye is None else tracking.right_eye,                
                np.zeros_like(preproc.image_processed), 
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype = self.tracking_param.dtype
        )

        return res
    
    def track(
            self,
            image: NDArray, 
            background_image: Optional[NDArray] = None,
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:

        if background_image is None:
            background_image = np.zeros_like(image)
            
        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )

        preproc = self.preprocess(image, background_image, centroid_in_input) 
        if preproc is None:
            return self.return_prediction_if_tracking_failed(
                preproc, #TODO this is None, make sure its p
                T_input_to_global,
                T_global_to_input,
            )
        
        tracking_resized = self.track_resized(preproc)
        if tracking_resized is None:
            return self.return_prediction_if_tracking_failed(
                preproc,
                T_input_to_global,
                T_global_to_input,
            )
        
        # kalman filter
        tracking, mask = tracking_resized
        self.kalman_filter.predict()
        measurement = self.tracking_to_measurement(tracking)
        self.kalman_filter.update(measurement)
        self.prediction_to_tracking(tracking)

        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )

        res = np.array(
            (
                True,
                np.zeros(1,dtype=DTYPE_EYE) if tracking.left_eye is None else tracking.left_eye, 
                np.zeros(1,dtype=DTYPE_EYE) if tracking.right_eye is None else tracking.right_eye,                
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype = self.tracking_param.dtype
        )

        return res

    