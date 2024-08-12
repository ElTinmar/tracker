from numpy.typing import NDArray
from typing import Optional
from .core import TailTracker, TailTracking
from .utils import tail_skeleton_ball
from tracker.prepare_image import prepare_image

class TailTracker_CPU(TailTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray] # TODO maybe provide a transformation from local to global coordinates
        ) -> Optional[TailTracking]:
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
            vertical_offset_px=self.tracking_param.crop_offset_tail_px,
            centroid=centroid,
            contrast=self.tracking_param.tail_contrast,
            gamma=self.tracking_param.tail_gamma,
            brightness=self.tracking_param.tail_brightness,
            blur_sz_px=self.tracking_param.blur_sz_px,
            median_filter_sz_px=self.tracking_param.median_filter_sz_px
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

        res = TailTracking(
            num_tail_pts = self.tracking_param.n_tail_points,
            num_tail_interp_pts = self.tracking_param.n_pts_interp,
            im_tail_shape = image_processed.shape,
            im_tail_fullres_shape = image_crop.shape,
            centroid = centroid,
            origin = origin,
            skeleton = skeleton,
            skeleton_interp = skeleton_interp,
            image = image_processed,
            image_fullres = image_crop
        )    

        return res
