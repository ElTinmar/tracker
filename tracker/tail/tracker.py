from numpy.typing import NDArray
import numpy as np
from typing import Optional
from .core import TailTracker
from .utils import tail_skeleton_ball
from tracker.prepare_image import preprocess_image
from geometry import transform2d, Affine2DTransform

class TailTracker_CPU(TailTracker):

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

        failed = np.zeros((1,), dtype=self.tracking_param.dtype)

        if (image is None) or (image.size == 0) or (centroid is None):
            return failed
        
        preproc = preprocess_image(image, centroid, self.tracking_param)
        
        if preproc is None:
            return failed

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
        skeleton_cropped = transform2d(preproc.resize_transform, skeleton_resized)
        skeleton_input = transform2d(preproc.crop_transform, skeleton_cropped)
        skeleton_global = transform2d(transformation_matrix, skeleton_input)

        skeleton_interp_cropped = transform2d(preproc.resize_transform, skeleton_interp_resized)
        skeleton_interp_input = transform2d(preproc.crop_transform, skeleton_interp_cropped)
        skeleton_interp_global = transform2d(transformation_matrix, skeleton_interp_input)

        # save result to numpy structured array
        res = np.array(
            (
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
                preproc.image_crop
            ), 
            dtype= self.tracking_param.dtype
        )

        return res
