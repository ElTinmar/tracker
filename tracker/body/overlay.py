from image_tools import im2rgb, im2uint8
from geometry import to_homogeneous, from_homogeneous
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import BodyOverlay, BodyTracking

class BodyOverlay_opencv(BodyOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[BodyTracking], 
            transformation_matrix: NDArray 
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish bounding box top left coordinates
        '''

        if (tracking is not None) and (tracking.centroid is not None):

            overlay = im2rgb(im2uint8(image))
            
            src = tracking.centroid
            heading = self.overlay_param.heading_len_px * tracking.heading[:,0]
            dst = src + heading

            # compute transformation
            pts = np.vstack((src, dst))
            pts_ = from_homogeneous((transformation_matrix @ to_homogeneous(pts).T).T)

            # heading
            overlay = cv2.line(
                overlay,
                pts_[0].astype(np.int32),
                pts_[1].astype(np.int32),
                self.overlay_param.heading_color_BGR,
                self.overlay_param.thickness
            )

            # show heading direction with a circle (easier than arrowhead)
            filled = -1
            overlay = cv2.circle(
                overlay,
                pts_[1].astype(np.int32),
                self.overlay_param.arrow_radius_px,
                self.overlay_param.heading_color_BGR,
                filled
            )
        
            return overlay
