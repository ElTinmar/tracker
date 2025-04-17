from numpy.typing import NDArray
import numpy as np
from typing import Optional, Tuple
from .core import TailTracker
from .utils import tail_skeleton_ball
from tracker.prepare_image import preprocess_image
from geometry import SimilarityTransform2D
from filterpy.common import kinematic_kf

class TailTracker_CPU(TailTracker):

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

        # track
        skeleton_resized, skeleton_interp_resized = tail_skeleton_ball(
            image = preproc.image_processed,
            ball_radius_px = self.tracking_param.ball_radius_px,
            arc_angle_deg = self.tracking_param.arc_angle_deg,
            tail_length_px = self.tracking_param.tail_length_px,
            n_tail_points = self.tracking_param.n_tail_points,
            n_pts_arc = self.tracking_param.n_pts_arc,
            n_pts_interp = self.tracking_param.n_pts_interp,
            w = self.tracking_param.resized_dimension_px[0] 
        )

        # transform coordinates
        skeleton_cropped = preproc.T_resized_to_cropped.transform_points(skeleton_resized)
        skeleton_input = preproc.T_cropped_to_input.transform_points(skeleton_cropped)
        skeleton_global = T_input_to_global.transform_points(skeleton_input)

        skeleton_interp_cropped = preproc.T_resized_to_cropped.transform_points(skeleton_interp_resized)
        skeleton_interp_input = preproc.T_cropped_to_input.transform_points(skeleton_interp_cropped)
        skeleton_interp_global = T_input_to_global.transform_points(skeleton_interp_input)

        pix_per_mm_global = self.tracking_param.pix_per_mm
        pix_per_mm_input = pix_per_mm_global * T_global_to_input.scale_factor
        pix_per_mm_cropped = pix_per_mm_input * preproc.T_input_to_cropped.scale_factor
        pix_per_mm_resized = pix_per_mm_cropped * preproc.T_cropped_to_resized.scale_factor
        
        # save result to numpy structured array
        res = np.array(
            (
                True,
                self.tracking_param.n_tail_points,
                self.tracking_param.n_pts_interp,
                centroid, 
                skeleton_resized,
                skeleton_cropped,
                skeleton_input,
                skeleton_global,
                skeleton_interp_resized,
                skeleton_interp_cropped,
                skeleton_interp_input,
                skeleton_interp_global,
                preproc.image_processed,
                preproc.image_cropped,
                pix_per_mm_global,
                pix_per_mm_input,
                pix_per_mm_cropped,
                pix_per_mm_resized,
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
        self.N_DIM = 2 * self.tracking_param.n_pts_interp
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
            measurement[0:self.N_DIM,0] = tracking['skeleton_interp_resized'].flatten()
        else:
            measurement = None

        return measurement

    def prediction_to_tracking(self, tracking: NDArray) -> None:
        '''Side effect: modify tracking in-place'''
        
        # TODO do that for resized, cropped, input and global
        tracking['skeleton_interp_resized'] = self.kalman_filter.x[0:self.N_DIM,0].reshape((self.tracking_param.n_pts_interp,2))


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