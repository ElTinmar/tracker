from image_tools import im2rgb, im2uint8
from geometry import SimilarityTransform2D

import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import BodyOverlay

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

    def overlay_global(
            self,
            image: NDArray, 
            tracking: Optional[NDArray],
            T_global_to_input: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Optional[NDArray]:

        if tracking is None:
            return None
            
        return self._overlay(
            centroid = tracking['centroid_global'],
            body_axes = tracking['body_axes_global'],
            image = image,
            pix_per_mm = tracking['pix_per_mm_global'],
            transformation = T_global_to_input
        )
    
    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:

        if tracking is None:
            return None
        
        return self._overlay(
            image = tracking['image_cropped'],
            centroid = tracking['centroid_cropped'],
            body_axes = tracking['body_axes'],
            pix_per_mm = tracking['pix_per_mm_cropped'],
        )

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        
        if tracking is None:
            return None

        return self._overlay(
            image = tracking['image_processed'],
            centroid = tracking['centroid_resized'],
            body_axes = tracking['body_axes'],
            pix_per_mm = tracking['pix_per_mm_resized'],
        )
            
    def _overlay(
            self,
            image: NDArray, 
            centroid: NDArray,
            body_axes: NDArray, 
            pix_per_mm: float,
            transformation: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Optional[NDArray]:


        overlay = im2rgb(im2uint8(image))
        original = overlay.copy()        

        pix_per_mm_input =  pix_per_mm * transformation.scale_factor
        heading_len_px = max(1,int(self.overlay_param.heading_len_mm * pix_per_mm_input))
        arrow_radius_px = max(1,int(self.overlay_param.arrow_radius_mm * pix_per_mm_input))

        front = heading_len_px * body_axes[:,0]
        right = heading_len_px/2 * body_axes[:,1]
        yy = centroid + front
        xx = centroid + right

        # compute transformation
        pts = np.vstack((centroid, xx, yy))
        pts_ = transformation.transform_points(pts)

        overlay = draw_arrow(
            overlay, 
            pts_[0],
            pts_[2], 
            self.overlay_param.heading_color_BGR,
            self.overlay_param.thickness,
            arrow_radius_px
        )

        overlay = draw_arrow(
            overlay, 
            pts_[0],
            pts_[1], 
            self.overlay_param.lateral_color_BGR,
            self.overlay_param.thickness,
            arrow_radius_px
        )

        overlay = cv2.addWeighted(
            overlay, self.overlay_param.alpha, 
            original, 1 - self.overlay_param.alpha, 
            0
        )
        
        return overlay
