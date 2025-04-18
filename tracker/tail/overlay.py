import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import im2uint8, im2rgb
from geometry import SimilarityTransform2D
from .core import TailOverlay

class TailOverlay_opencv(TailOverlay):

    def overlay_global(
            self,
            image: NDArray, 
            tracking: Optional[NDArray],
            T_global_to_input: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Optional[NDArray]:

        if tracking is None:
            return None
            
        return self._overlay(
            skeleton = tracking['skeleton_global'],
            skeleton_interp = tracking['skeleton_interp_global'],
            image = image,
            pix_per_mm = tracking['pix_per_mm_global'],
            transformation = T_global_to_input
        )
    
    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:

        if tracking is None:
            return None
        
        return self._overlay(
            image = tracking['image_cropped'],
            skeleton = tracking['skeleton_cropped'],
            skeleton_interp = tracking['skeleton_interp_cropped'],
            pix_per_mm = tracking['pix_per_mm_cropped'],
        )

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        
        if tracking is None:
            return None

        return self._overlay(
            image = tracking['image_processed'],
            skeleton = tracking['skeleton_resized'],
            skeleton_interp = tracking['skeleton_interp_resized'],
            pix_per_mm = tracking['pix_per_mm_resized'],
        )
    
    def _overlay(
            self,
            image: NDArray, 
            skeleton: NDArray,
            skeleton_interp: NDArray,
            pix_per_mm: float,
            transformation: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:
                
        overlay = im2rgb(im2uint8(image))
        original = overlay.copy()        

        pix_per_mm_input =  pix_per_mm * transformation.scale_factor
        ball_radius_px = max(1,int(self.overlay_param.ball_radius_mm * pix_per_mm_input))
            
        transformed_coord_interp = transformation.transform_points(skeleton_interp)
        tail_segments = zip(transformed_coord_interp[:-1,], transformed_coord_interp[1:,])
        for pt1, pt2 in tail_segments:
            overlay = cv2.line(
                overlay,
                pt1.astype(np.int32),
                pt2.astype(np.int32),
                self.overlay_param.color_BGR,
                self.overlay_param.thickness
            )

        transformed_coord = transformation.transform_points(skeleton)
        for pt in transformed_coord:
            overlay = cv2.circle(
                overlay, 
                pt.astype(np.int32), 
                ball_radius_px, 
                self.overlay_param.color_BGR, 
                1
            )

        overlay = cv2.addWeighted(
            overlay, self.overlay_param.alpha, 
            original, 1 - self.overlay_param.alpha, 
            0
        )

        return overlay
