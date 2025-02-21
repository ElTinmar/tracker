from numpy.typing import NDArray
import numpy as np
from typing import Optional
from .core import TailTracker
from .utils import tail_skeleton_ball
from tracker.prepare_image import crop, resize
from image_tools import enhance

class TailTracker_CPU(TailTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] # TODO maybe provide a transformation from local to global coordinates and store both in result
        ) -> Optional[NDArray]:
        """
        output coordinates: 
            - (0,0) = fish centroid
            - scale of the full-resolution image, before resizing
        """

        if (image is None) or (image.size == 0) or (centroid is None):
            return None
        
        # pre-process image: crop/resize/tune intensity
        (origin, image_crop) = crop(
            image = image,
            source_crop_dimension_px = self.tracking_param.source_crop_dimension_px,
            vertical_offset_px=self.tracking_param.crop_offset_tail_px,
            centroid = centroid
        )

        image_resized = resize(
            image = image_crop,
            target_crop_dimension_px = self.tracking_param.crop_dimension_px, 
        )

        image_processed = enhance(
            image_resized,
            contrast = self.tracking_param.tail_contrast,
            gamma = self.tracking_param.tail_gamma,
            brightness = self.tracking_param.tail_brightness,
            blur_sz_px = self.tracking_param.blur_sz_px,
            median_filter_sz_px = self.tracking_param.median_filter_sz_px
        )

        skeleton, skeleton_interp = tail_skeleton_ball(
            image = image_processed,
            ball_radius_px = self.tracking_param.ball_radius_px,
            arc_angle_deg = self.tracking_param.arc_angle_deg,
            tail_length_px = self.tracking_param.tail_length_px,
            n_tail_points = self.tracking_param.n_tail_points,
            n_pts_arc = self.tracking_param.n_pts_arc,
            dist_swim_bladder_px = self.tracking_param.dist_swim_bladder_px,
            n_pts_interp = self.tracking_param.n_pts_interp,
            origin = origin*self.tracking_param.resize,
            resize = self.tracking_param.resize,
            w = self.tracking_param.crop_dimension_px[0]
        )

        # save result to numpy structured array
        res = np.array(
            (
                skeleton is None,
                self.tracking_param.n_tail_points,
                self.tracking_param.n_pts_interp,
                np.zeros((1,2), np.float32) if centroid is None else centroid, 
                np.zeros((1,2), np.float32) if origin is None else origin,
                np.zeros((self.tracking_param.n_tail_points,2), np.float32) if skeleton is None else skeleton,
                np.zeros((self.tracking_param.n_pts_interp,2), np.float32) if skeleton_interp is None else skeleton_interp,
                image_processed,
                image_crop
            ), 
            dtype= self.tracking_param.dtype()
        )

        return res
