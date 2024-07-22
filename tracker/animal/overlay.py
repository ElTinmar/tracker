from image_tools import im2uint8, im2rgb
from geometry import to_homogeneous, Affine2DTransform
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Optional
from .core import AnimalOverlay, AnimalTracking

class AnimalOverlay_opencv(AnimalOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[AnimalTracking], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> Optional[NDArray]:

        if (tracking is not None) and (tracking.centroids.size > 0):

            overlay = im2rgb(im2uint8(image))
            original = overlay.copy()        

            for idx, id in zip(tracking.indices, tracking.identities):

                # draw centroid
                x,y,_ = transformation_matrix @ to_homogeneous(tracking.centroids[idx,:])
                overlay = cv2.circle(
                    overlay,
                    (int(x),int(y)), 
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
        
