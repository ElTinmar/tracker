import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass
import cv2
from .animal import AnimalTracking
from image_tools import im2rgb

@dataclass
class AnimalTrackerParamOverlay:
    pix_per_mm: float = 40.0
    radius_mm: float = 0.1
    centroid_color: tuple = (255, 128, 128)
    bbox_color:tuple = (255, 255, 255) 
    centroid_thickness: int = -1
    bbox_thickness: int = 2

    def mm2px(self, val_mm):
        return int(val_mm * self.pix_per_mm) 

    @property
    def radius_px(self):
        return self.mm2px(self.radius_mm)

def overlay(
        image: NDArray, 
        tracking: Optional[AnimalTracking], 
        param: AnimalTrackerParamOverlay,
        scale: float
    ) -> Optional[NDArray]:

    if tracking is not None:

        overlay = im2rgb(image)

        # draw centroid
        for (x,y) in tracking.centroids*scale:
            overlay = cv2.circle(
                overlay,
                (int(x),int(y)), 
                param.radius_px, 
                param.centroid_color, 
                param.centroid_thickness
            )

        # draw bounding boxes
        for (left, bottom, right, top) in tracking.bounding_boxes*scale:
            overlay = cv2.rectangle(
                overlay, 
                (int(left), int(top)),
                (int(right), int(bottom)), 
                param.bbox_color, 
                param.bbox_thickness
            )

        return overlay

