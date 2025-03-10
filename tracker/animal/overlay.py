from image_tools import im2uint8, im2rgb
from geometry import transform2d, Affine2DTransform
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalOverlay

# TODO maybe one function for input / cropped / resized space?
# no need to pass image as input, use image in tracking

class AnimalOverlay_opencv(AnimalOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[NDArray], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> Optional[NDArray]:

        if tracking is not None:

            overlay = im2rgb(im2uint8(image))
            original = overlay.copy()        

            for centroid in tracking['centroids_global']:

                # draw centroid
                x,y = transform2d(transformation_matrix, centroid)
                
                overlay = cv2.circle(
                    overlay,
                    (int(centroid[0]),int(centroid[1])), 
                    self.overlay_param.radius_px, 
                    self.overlay_param.centroid_color_BGR, 
                    self.overlay_param.centroid_thickness
                )
                
                # show ID
                cv2.putText(
                    overlay, 
                    str(id), (int(x+self.overlay_param.label_offset), int(y-self.overlay_param.label_offset)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                    self.overlay_param.id_str_color_BGR, 
                    2, 
                    cv2.LINE_AA
                )
            
            overlay = cv2.addWeighted(overlay, self.overlay_param.alpha, original, 1 - self.overlay_param.alpha, 0)

            return overlay
        
