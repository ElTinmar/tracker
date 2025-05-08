from numpy.typing import NDArray
import numpy as np
from typing import Optional, Tuple
from .core import TailTracker
from .utils import tail_skeleton_ball, interpolate_skeleton
from tracker.prepare_image import preprocess_image, Preprocessing
from tracker.core import Resolution
from geometry import SimilarityTransform2D
from filterpy.common import kinematic_kf

class Tracking:
    def __init__(
            self, 
            num_pts: int, 
            num_pts_interp: int
        ) -> None:
        
        self.num_pts = num_pts
        self.num_pts_interp = num_pts_interp
        self.angles = np.zeros((num_pts,), np.float32)
        self.skeleton_resized = np.zeros((num_pts,2), np.float32)
        self.skeleton_cropped = np.zeros((num_pts,2), np.float32)
        self.skeleton_input = np.zeros((num_pts,2), np.float32)
        self.skeleton_global = np.zeros((num_pts,2), np.float32)
        self.skeleton_interp_resized = np.zeros((num_pts_interp,2), np.float32)
        self.skeleton_interp_cropped = np.zeros((num_pts_interp,2), np.float32)
        self.skeleton_interp_input = np.zeros((num_pts_interp,2), np.float32)
        self.skeleton_interp_global = np.zeros((num_pts_interp,2), np.float32)

class TailTracker_CPU(TailTracker):

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
        centroid: Optional[NDArray] = None, # centroids in input space
        ) -> Optional[Preprocessing]:
        
        return preprocess_image(image, centroid, self.tracking_param)

    def track_resized(
            self,
            preproc: Preprocessing, 
        ) -> Tracking:
        
        tracking = Tracking(self.tracking_param.n_tail_points, self.tracking_param.n_pts_interp)
        tracking.skeleton_resized, tracking.angles = tail_skeleton_ball(
            image = preproc.image_processed,
            ball_radius_px = self.tracking_param.ball_radius_px,
            arc_angle_deg = self.tracking_param.arc_angle_deg,
            tail_length_px = self.tracking_param.tail_length_px,
            n_tail_points = self.tracking_param.n_tail_points,
            n_pts_arc = self.tracking_param.n_pts_arc,
            w = self.tracking_param.resized_dimension_px[0] 
        )
        tracking.skeleton_interp_resized = interpolate_skeleton(
            tracking.skeleton_resized,
            n_pts_interp = self.tracking_param.n_pts_interp,
        )
        return tracking

    def transform_coordinate_system(
            self,
            tracking: Tracking,        
            preproc: Preprocessing,
            T_input_to_global,
            T_global_to_input
        ) -> Resolution:
        '''modifies tracking in-place with side-effect and returns resolution'''
        
        tracking.skeleton_cropped = preproc.T_resized_to_cropped.transform_points(tracking.skeleton_resized)
        tracking.skeleton_input = preproc.T_cropped_to_input.transform_points(tracking.skeleton_cropped)
        tracking.skeleton_global = T_input_to_global.transform_points(tracking.skeleton_input)

        tracking.skeleton_interp_cropped = preproc.T_resized_to_cropped.transform_points(tracking.skeleton_interp_resized)
        tracking.skeleton_interp_input = preproc.T_cropped_to_input.transform_points(tracking.skeleton_interp_cropped)
        tracking.skeleton_interp_global = T_input_to_global.transform_points(tracking.skeleton_interp_input)

        resolution = Resolution()
        resolution.pix_per_mm_global = self.tracking_param.pix_per_mm
        resolution.pix_per_mm_input = resolution.pix_per_mm_global * T_global_to_input.scale_factor
        resolution.pix_per_mm_cropped = resolution.pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        resolution.pix_per_mm_resized = resolution.pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor
        
        return resolution

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

        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )
        
        preproc = self.preprocess(image, centroid_in_input)
        if preproc is None:
            return self.tracking_param.failed

        tracking = self.track_resized(preproc)
        resolution = self.transform_coordinate_system(
            tracking, 
            preproc, 
            T_input_to_global, 
            T_global_to_input
        )

        # save result to numpy structured array
        res = np.array(
            (
                True,
                self.tracking_param.n_tail_points,
                self.tracking_param.n_pts_interp,
                centroid_in_input, 
                tracking.skeleton_resized,
                tracking.skeleton_cropped,
                tracking.skeleton_input,
                tracking.skeleton_global,
                tracking.skeleton_interp_resized,
                tracking.skeleton_interp_cropped,
                tracking.skeleton_interp_input,
                tracking.skeleton_interp_global,
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype= self.tracking_param.dtype
        )

        return res


class TailTrackerKalman(TailTracker_CPU):
    # TODO this does not respect segment length
    # model the angle for each segment instead

    def __init__(
            self, 
            fps: int, 
            model_order: int, 
            model_uncertainty: float = 1.0,
            measurement_uncertainty: float = 1.0,
            *args, 
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.N_DIM = 2 * self.tracking_param.n_tail_points
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
        # TODO: kalman filter skeleton angles instead and compute skeleton_resized from angles
        
        measurement = np.zeros((self.N_DIM,1))
        measurement[0:self.N_DIM,0] = tracking.skeleton_resized.flatten()

        return measurement

    def prediction_to_tracking(self, tracking: Tracking) -> None:
        '''Side effect: modify tracking in-place'''
        
        # TODO do that for resized, cropped, input and global
        tracking.skeleton_resized = self.kalman_filter.x[0:self.N_DIM,0].reshape((self.tracking_param.n_tail_points,2))
        tracking.skeleton_interp_resized = interpolate_skeleton(
            tracking.skeleton_resized,
            n_pts_interp = self.tracking_param.n_pts_interp,
        )

    def return_prediction_if_tracking_failed(
            self,
            centroid,
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
                self.tracking_param.n_tail_points,
                self.tracking_param.n_pts_interp,
                centroid, 
                tracking.skeleton_resized,
                tracking.skeleton_cropped,
                tracking.skeleton_input,
                tracking.skeleton_global,
                tracking.skeleton_interp_resized,
                tracking.skeleton_interp_cropped,
                tracking.skeleton_interp_input,
                tracking.skeleton_interp_global,
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype= self.tracking_param.dtype
        )

        return res

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] = None, # centroids in global space
            T_input_to_global: Optional[SimilarityTransform2D] = SimilarityTransform2D.identity()
        ) -> NDArray:

        self.tracking_param.input_image_shape = image.shape

        centroid_in_input, T_global_to_input = self.transform_input_centroid(
            centroid,
            T_input_to_global
        )
        
        preproc = self.preprocess(image, centroid_in_input)
        if preproc is None:
            return self.return_prediction_if_tracking_failed(
                preproc, # This is None, make sure its ok
                centroid_in_input,
                T_input_to_global,
                T_global_to_input,
            )

        tracking = self.track_resized(preproc)

        # kalman filter
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
                self.tracking_param.n_tail_points,
                self.tracking_param.n_pts_interp,
                centroid_in_input, 
                tracking.skeleton_resized,
                tracking.skeleton_cropped,
                tracking.skeleton_input,
                tracking.skeleton_global,
                tracking.skeleton_interp_resized,
                tracking.skeleton_interp_cropped,
                tracking.skeleton_interp_input,
                tracking.skeleton_interp_global,
                preproc.image_processed,
                preproc.image_cropped,
                resolution.pix_per_mm_global,
                resolution.pix_per_mm_input,
                resolution.pix_per_mm_cropped,
                resolution.pix_per_mm_resized,
            ), 
            dtype= self.tracking_param.dtype
        )

        return res
    