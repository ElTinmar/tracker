from image_tools import  bwareafilter_props_cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .core import BodyTracker
from .utils import get_orientation, get_best_centroid_index
from tracker.prepare_image import preprocess_image
from geometry import SimilarityTransform2D
import cv2
from filterpy.kalman import KalmanFilter
from enum import Enum

class BodyTracker_CPU(BodyTracker):
        
    def track(
            self,
            image: NDArray, 
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
        
        self.tracking_param.input_image_shape = image.shape

        T_global_to_input = T_input_to_global.inv()
        
        if centroid is None:
            centroid_input = None
        else:
            centroid_input = T_global_to_input.transform_points(centroid).squeeze()

        preproc = preprocess_image(image, centroid_input, self.tracking_param)
            
        if preproc is None:
            return self.tracking_param.failed

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
            return self.tracking_param.failed
        
        centroids_resized = np.array([[blob.centroid[1], blob.centroid[0]] for blob in props]) #(row, col) to (x,y)
        centroids_cropped = preproc.T_resized_to_cropped.transform_points(centroids_resized)
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
            return self.tracking_param.failed
        
        body_axes_global = T_input_to_global.transform_vectors(body_axes) 
        
        angle_rad = np.arctan2(body_axes[1,1], body_axes[0,1])
        angle_rad_global = np.arctan2(body_axes_global[1,1], body_axes_global[0,1])

        pix_per_mm_global = self.tracking_param.pix_per_mm
        pix_per_mm_input = pix_per_mm_global * T_global_to_input.scale_factor
        pix_per_mm_cropped = pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        pix_per_mm_resized = pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor
        
        res = np.array(
            (
                True,
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
    
class MotionModel(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    CONSTANT_JERK = "constant_jerk"

def normalize_angle(theta: float) -> float:
    return np.arctan2(np.sin(theta), np.cos(theta))

def angular_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

class BodyTrackerKalman(BodyTracker_CPU):

    def __init__(
            self, 
            fps: int, 
            model: MotionModel = MotionModel.CONSTANT_VELOCITY, 
            *args, 
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.fps = fps
        dt = 1/fps
        
        if model == MotionModel.CONSTANT_VELOCITY:
            # state = x,y,theta + first time derivatives
            self.kalman_filter = KalmanFilter(dim_x=6, dim_z=3) 
            self.kalman_filter.x = np.zeros((6,1)) # initial state
            self.kalman_filter.F = np.array([
                [1,  0,  0, dt,  0,  0],
                [0,  1,  0,  0, dt,  0],
                [0,  0,  1,  0,  0, dt],
                [0,  0,  0,  1,  0,  0],
                [0,  0,  0,  0,  1,  0],
                [0,  0,  0,  0,  0,  1],
            ])
            self.kalman_filter.H = np.array([
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0]
            ])
            self.kalman_filter.P = 100 * np.eye(6) # state uncertainty
            self.kalman_filter.Q = np.eye(6) # model uncertainty

        elif model == MotionModel.CONSTANT_ACCELERATION:   
            # state = x,y,theta + first and second time derivatives
            self.kalman_filter = KalmanFilter(dim_x=9, dim_z=3) 
            self.kalman_filter.F = np.array([
                [1, 0, 0, dt, 0, 0, dt**2/2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, dt**2/2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, dt**2/2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            self.kalman_filter.x = np.zeros((9,1)) # initial state
            self.kalman_filter.H = np.array([
                [1,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0]
            ])
            self.kalman_filter.P = 100 * np.eye(9) # state uncertainty
            self.kalman_filter.Q = np.diag([1,1,1, 1,1,1, 1,1,1]) # model uncertainty

        elif model == MotionModel.CONSTANT_JERK:   
            # state = x,y,theta + first and second time derivatives
            self.kalman_filter = KalmanFilter(dim_x=12, dim_z=3) 
            self.kalman_filter.F = np.array([
                [1, 0, 0, dt, 0, 0, dt**2/2, 0, 0, dt**3/6, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, dt**2/2, 0, 0, dt**3/6, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, dt**2/2, 0, 0, dt**3/6],
                [0, 0, 0, 1, 0, 0, dt, 0, 0, dt**2/2, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0, 0, dt**2/2, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, dt**2/2],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            self.kalman_filter.x = np.zeros((12,1)) # initial state
            #self.kalman_filter.x[2] = -np.pi
            self.kalman_filter.H = np.array([
                [1,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0,0]
            ])
            self.kalman_filter.P = 100 * np.eye(12) # state uncertainty
            self.kalman_filter.Q = np.diag([1,1,1, 1,1,1, 1,1,1, 1,1,1]) # model uncertainty
        
        self.kalman_filter.R = np.diag([1,1,1]) # measurement uncertainty

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: Optional[SimilarityTransform2D] = SimilarityTransform2D.identity()
        ) -> NDArray:

        tracking = super().track(image, centroid, T_input_to_global)

        self.kalman_filter.predict()

        if tracking['success']:
            measurement = np.zeros((3,1))
            measurement[:2,0] = tracking['centroid_resized']
            measurement[2] = tracking['angle_rad']
        

            angle_predicted = self.kalman_filter.x[2]
            delta = angular_difference(measurement[2], angle_predicted)
            if abs(delta) > np.pi / 2:
                measurement[2] += np.pi 
                measurement[2] = normalize_angle(measurement[2])

        else:
            measurement = None

        self.kalman_filter.update(measurement)

        # TODO do that for resized, cropped, input and global
        tracking['centroid_resized'] = self.kalman_filter.x[:2,0]
        tracking['angle_rad'] = self.kalman_filter.x[2]

        # TODO also need to update 'body_axes'
        tracking['body_axes'] = np.array([
            [-np.sin(tracking['angle_rad']), np.cos(tracking['angle_rad'])],
            [np.cos(tracking['angle_rad']), np.sin(tracking['angle_rad'])]
        ])
        
        return tracking