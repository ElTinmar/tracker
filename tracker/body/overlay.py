from image_tools import im2rgb, im2uint8
from geometry import to_homogeneous, from_homogeneous, Affine2DTransform

import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import BodyOverlay, BodyTracking

def draw_arrow(image, src, dst, color, thickness, radius):

    # heading
    image = cv2.line(
        image,
        src.astype(np.int32),
        dst.astype(np.int32),
        color,
        thickness
    )

    # show heading direction with a circle (easier than arrowhead)
    filled = -1
    image = cv2.circle(
        image,
        dst.astype(np.int32),
        radius,
        color,
        filled
    )

    return image


class BodyOverlay_opencv(BodyOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[BodyTracking], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish bounding box top left coordinates
        '''

        if (tracking is not None) and (tracking.centroid is not None):

            overlay = im2rgb(im2uint8(image))
            
            src = tracking.centroid
            front = self.overlay_param.heading_len_px * tracking.heading[:,0]
            right = self.overlay_param.heading_len_px/2 * tracking.heading[:,1]
            yy = src + front
            xx = src + right

            # compute transformation
            pts = np.vstack((src, xx, yy))
            pts_ = from_homogeneous((transformation_matrix @ to_homogeneous(pts).T).T)

            overlay = draw_arrow(
                overlay, 
                pts_[0],
                pts_[2], 
                self.overlay_param.heading_color_BGR,
                self.overlay_param.thickness,
                self.overlay_param.arrow_radius_px
            )

            overlay = draw_arrow(
                overlay, 
                pts_[0],
                pts_[1], 
                self.overlay_param.lateral_color_BGR,
                self.overlay_param.thickness,
                self.overlay_param.arrow_radius_px
            )
        
            return overlay
