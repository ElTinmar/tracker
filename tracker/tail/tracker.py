import math
from scipy.interpolate import splprep, splev
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import enhance, im2uint8
from .core import TailTracker, TailTracking
from .utils import tail_skeleton_ball

class TailTracker_CPU(TailTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray]
        ) -> Optional[TailTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
            return None

        if self.tracking_param.resize != 1:
            image = cv2.resize(
                image, 
                self.tracking_param.crop_dimension_px[::-1], 
                interpolation=cv2.INTER_NEAREST
            )

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        pad_width = np.max(self.tracking_param.crop_dimension_px)
        offset = np.array((-w//2, -h//2+self.tracking_param.crop_offset_tail_px), dtype=np.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(np.int32) + offset + np.array([pad_width,pad_width]) 
        right, top = left+w, bottom+h 

        # pad image to get fixed image size
        image_padded = np.pad(image, (pad_width,pad_width))

        image_crop = image_padded[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance(
            image_crop,
            self.tracking_param.tail_contrast,
            self.tracking_param.tail_gamma,
            self.tracking_param.tail_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        skeleton, skeleton_interp = tail_skeleton_ball(
            image_crop = image_crop,
            ball_radius_px = self.tracking_param.ball_radius_px,
            arc_angle_deg = self.tracking_param.arc_angle_deg,
            tail_length_px = self.tracking_param.tail_length_px,
            n_tail_points = self.tracking_param.n_tail_points,
            n_pts_arc = self.tracking_param.n_pts_arc,
            dist_swim_bladder_px = self.tracking_param.dist_swim_bladder_px,
            n_pts_interp = self.tracking_param.n_pts_interp,
            offset = offset,
            resize = self.tracking_param.resize,
            w = w
        )

        res = TailTracking(
            num_tail_pts = self.tracking_param.n_tail_points,
            num_tail_interp_pts = self.tracking_param.n_pts_interp,
            im_tail_shape = image_crop.shape,
            centroid = centroid,
            offset = offset,
            skeleton = skeleton,
            skeleton_interp = skeleton_interp,
            image = image_crop
        )    

        return res
