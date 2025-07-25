from image_tools import filter_contours, filter_connected_comp,  im2gray
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from .core import BodyTracker, BodyTrackerParamTracking
from .utils import detect_flips, get_best_centroid_index
from tracker.prepare_image import preprocess_image, Preprocessing
from tracker.core import Resolution
from tracker.kalman_utils import kalman_update_wrap_angle
from geometry import SimilarityTransform2D
import cv2
from filterpy.common import kinematic_kf
from collections import deque
from dataclasses import dataclass

@dataclass
class Tracking:
    centroid_resized: NDArray = np.zeros((2,), np.float32)
    centroid_cropped: NDArray = np.zeros((2,), np.float32)
    centroid_input: NDArray = np.zeros((2,), np.float32)
    centroid_global: NDArray = np.zeros((2,), np.float32)
    body_axes: NDArray = np.zeros((2,2), np.float32)
    body_axes_global: NDArray = np.zeros((2,2), np.float32)
    angle_rad: float = 0.0
    angle_rad_global: float = 0.0

class BodyTracker_CPU(BodyTracker):

    def __init__(
        self, 
        fps: Optional[float] = None, 
        history_sec: float = 0.2,
        *args, 
        **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)

        self.heading_history = None
        if fps is not None:
            self.fps = fps
            self.history_sec = history_sec
            self.heading_history = deque(maxlen=int(history_sec*fps))

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
        ) -> Optional[Tuple[Preprocessing, NDArray]]:
        
        preproc = preprocess_image(image, background_image, centroid, self.tracking_param)

        if preproc is None:
            return None
            
        centroid_cropped = preproc.T_input_to_cropped.transform_points(centroid).squeeze()
        centroid_resized = preproc.T_cropped_to_resized.transform_points(centroid_cropped).squeeze()

        return preproc, centroid_resized

    def track_resized(
            self,
            preproc: Preprocessing, 
            centroid_in_resized: Optional[NDArray] = None, # centroids in resized space
        ) -> Optional[Tuple[Tracking, NDArray]]:

        mask = cv2.compare(
            preproc.image_processed, 
            self.tracking_param.intensity, # type: ignore
            cv2.CMP_GT
        )
        blobs = filter_connected_comp(
            mask, 
            min_size = self.tracking_param.min_size_px,
            max_size = self.tracking_param.max_size_px, 
            min_length = None if self.tracking_param.min_length_px == 0 else self.tracking_param.min_length_px,
            max_length = None if self.tracking_param.max_length_px == 0 else self.tracking_param.max_length_px,
            min_width = None if self.tracking_param.min_width_px == 0 else self.tracking_param.min_width_px,
            max_width = None if self.tracking_param.max_width_px == 0 else self.tracking_param.max_width_px
        )

        if not blobs:
            return None
        
        tracking = Tracking()
        
        centroids_resized = np.array([blob.centroid for blob in blobs]) 
        index = get_best_centroid_index(centroids_resized, centroid_in_resized)

        tracking.centroid_resized = blobs[index].centroid
        tracking.body_axes = detect_flips(blobs[index].axes, self.heading_history)
        tracking.angle_rad = blobs[index].angle_rad

        return tracking, mask
        
    def transform_coordinate_system(
            self,
            tracking: Tracking,        
            preproc: Preprocessing,
            T_input_to_global,
            T_global_to_input
        ) -> Resolution:
        '''modifies tracking in-place with side-effect and returns resolution'''
        
        tracking.centroid_cropped = preproc.T_resized_to_cropped.transform_points(tracking.centroid_resized)
        tracking.centroid_input = preproc.T_cropped_to_input.transform_points(tracking.centroid_cropped)
        tracking.centroid_global = T_input_to_global.transform_points(tracking.centroid_input)
        tracking.body_axes_global = T_input_to_global.transform_vectors(tracking.body_axes) 
        tracking.angle_rad_global = np.arctan2(tracking.body_axes_global[1,1], tracking.body_axes_global[0,1])

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
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:
        '''
        centroid: centroid of the fish to track if it's already known.
        Useful when tracking multiple fish to discriminate between nearby blobs

        output coordinates: 
            - (0,0) = topleft corner of the bounding box
            - scale of the full-resolution image, before resizing
        '''
        self.tracking_param.input_image_dtype = image.dtype
        
        # only work with one channel
        image = im2gray(image)

        if background_image is None:
            background_image = np.zeros_like(image)
        background_image = im2gray(background_image)

        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )

        preprocessing = self.preprocess(image, background_image, centroid_in_input)
        if preprocessing is None:
            return self.tracking_param.failed
        
        preproc, centroid_in_resized = preprocessing 
        tracking_resized = self.track_resized(preproc, centroid_in_resized)
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
                tracking.body_axes, 
                tracking.body_axes_global,
                tracking.centroid_resized,
                tracking.centroid_cropped,
                tracking.centroid_input,
                tracking.centroid_global,
                tracking.angle_rad,
                tracking.angle_rad_global,
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                preproc.background_image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype=self.tracking_param.dtype
        )
        return res

class BodyTrackerKalman(BodyTracker_CPU):

    N_DIM = 3

    def __init__(
            self, 
            model_order: int, 
            model_uncertainty: float = 0.2,
            measurement_uncertainty: float = 1.0,
            *args, 
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        
        if self.fps is None:
            raise ValueError('provide valid fps as argument')
        
        dt = 1/self.fps
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
        measurement[:2,0] = tracking.centroid_resized
        measurement[2,0] = tracking.angle_rad

        return measurement

    def prediction_to_tracking(self, tracking: Tracking) -> None:
        '''Side effect: modify tracking in-place'''
        
        tracking.centroid_resized = self.kalman_filter.x[:2,0]
        tracking.angle_rad = self.kalman_filter.x[2,0]
        tracking.body_axes = np.array([
            [np.sin(tracking.angle_rad), np.cos(tracking.angle_rad)],
            [-np.cos(tracking.angle_rad), np.sin(tracking.angle_rad)]
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
                tracking.body_axes, 
                tracking.body_axes_global,
                tracking.centroid_resized,
                tracking.centroid_cropped,
                tracking.centroid_input,
                tracking.centroid_global,
                tracking.angle_rad,
                tracking.angle_rad_global,
                np.zeros_like(preproc.image_processed), 
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype=self.tracking_param.dtype
        )
        return res

    def track(
            self,
            image: NDArray, 
            background_image: Optional[NDArray] = None,
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:
        
        self.tracking_param.input_image_dtype = image.dtype
        
        # only work with one channel
        image = im2gray(image)

        if background_image is None:
            background_image = np.zeros_like(image)
        background_image = im2gray(background_image)

        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )

        preprocessing = self.preprocess(image, background_image, centroid_in_input)
        if preprocessing is None:
            return self.return_prediction_if_tracking_failed(
                preproc, #FIXME this is not decalred here (ZebVR Issue #48)
                T_input_to_global,
                T_global_to_input,
            )
        
        preproc, centroid_in_resized = preprocessing 
        tracking_resized = self.track_resized(preproc, centroid_in_resized)
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
        kalman_update_wrap_angle(self.kalman_filter, measurement, 2)
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
                tracking.body_axes, 
                tracking.body_axes_global,
                tracking.centroid_resized,
                tracking.centroid_cropped,
                tracking.centroid_input,
                tracking.centroid_global,
                tracking.angle_rad,
                tracking.angle_rad_global,
                mask, 
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype=self.tracking_param.dtype
        )
        return res