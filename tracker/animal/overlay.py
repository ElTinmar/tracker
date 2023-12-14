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

            # draw centroid
            for x,y,_ in (transformation_matrix @ to_homogeneous(tracking.centroids).T).T:
                overlay = cv2.circle(
                    overlay,
                    (int(x),int(y)), 
                    self.overlay_param.radius_px, 
                    self.overlay_param.centroid_color, 
                    self.overlay_param.centroid_thickness
                )

            # draw bounding boxes
            for (left, bottom, right, top) in tracking.bounding_boxes:
                rect = np.array([[left, top],[right, bottom]])
                bbox = (transformation_matrix @ to_homogeneous(rect).T).T
                topleft = bbox[0,:-1].astype(int)
                bottomright = bbox[1,:-1].astype(int)
                overlay = cv2.rectangle(
                    overlay, 
                    topleft,
                    bottomright, 
                    self.overlay_param.bbox_color, 
                    self.overlay_param.bbox_thickness
                )

            return overlay
        
