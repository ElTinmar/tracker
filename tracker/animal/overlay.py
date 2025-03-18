from image_tools import im2uint8, im2rgb
from geometry import SimilarityTransform2D
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalOverlay

class AnimalOverlay_opencv(AnimalOverlay):

    def overlay_global(
            self,
            image: NDArray, 
            tracking: Optional[NDArray],
            T_global_to_input: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Optional[NDArray]:

        if tracking is None:
            return None
            
        return self._overlay(
            centroids = tracking['centroids_global'],
            image = image,
            pix_per_mm = tracking['pix_per_mm_global'],
            transformation = T_global_to_input
        )

    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:

        if tracking is None:
            return None
        
        S = SimilarityTransform2D.scaling(tracking['downsample_ratio'])

        return self._overlay(
            centroids = tracking['centroids_cropped'],
            image = tracking['image_downsampled'],
            pix_per_mm = tracking['pix_per_mm_downsampled'],
            transformation = S
        )

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        
        if tracking is None:
            return None
        
        return self._overlay(
            centroids = tracking['centroids_resized'],
            pix_per_mm = tracking['pix_per_mm_resized'],
            image = tracking['image_processed']
        )

    def _overlay(
            self,
            centroids: NDArray,
            image: NDArray, 
            pix_per_mm: float,
            transformation: SimilarityTransform2D = SimilarityTransform2D.identity(),
        ) -> NDArray:


        overlay = im2rgb(im2uint8(image))
        original = overlay.copy()        

        for idx, centroid in enumerate(centroids):

            # draw centroid
            x,y = transformation.transform_points(centroid).ravel()

            pix_per_mm_input =  pix_per_mm * transformation.scale_factor
            radius_px = max(1,int(self.overlay_param.radius_mm * pix_per_mm_input))
            font_height_px = max(1, int(self.overlay_param.id_str_height_mm * pix_per_mm_input))
            
            overlay = cv2.circle(
                overlay,
                (int(x),int(y)), 
                radius_px, 
                self.overlay_param.centroid_color_BGR, 
                self.overlay_param.centroid_thickness
            )
            
            font_size = cv2.getFontScaleFromHeight(
                self.overlay_param.font,
                font_height_px, 
                self.overlay_param.font_thickness
            )

            # show ID
            cv2.putText(
                overlay, 
                str(idx), (int(x+self.overlay_param.label_offset), int(y-self.overlay_param.label_offset)), 
                self.overlay_param.font, font_size, 
                self.overlay_param.id_str_color_BGR, 
                self.overlay_param.font_thickness, 
                self.overlay_param.font_line_type
            )
        
        overlay = cv2.addWeighted(
            overlay, self.overlay_param.alpha, 
            original, 1 - self.overlay_param.alpha, 
            0
        )

        return overlay
    
