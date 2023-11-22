from .body import BodyTracking
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass
import cv2
from image_tools import im2rgb

    
@dataclass
class BodyTrackerParamOverlay:
    pix_per_mm: float = 40.0
    heading_len_mm: float = 1.5
    heading_color: tuple = (0,128,255)
    thickness: int = 2

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px

    @property
    def heading_len_px(self):
        return self.mm2px(self.heading_len_mm)

def overlay(
        image: NDArray, 
        tracking: BodyTracking, 
        param: BodyTrackerParamOverlay,
        translation_vec: Optional[NDArray],
        scale: float
    ) -> Optional[NDArray]:
    '''
    translation_vec: if tracking on cropped image, translation_vec of cropped part in larger image
    '''

    if tracking is not None:

        overlay = im2rgb(image)

        pt1 = tracking.centroid * scale
        if translation_vec is not None:
            pt1 = pt1 + translation_vec
        pt2 = pt1 + param.heading_len_px * tracking.heading[:,0]

        # heading
        overlay = cv2.line(
            overlay,
            pt1.astype(np.int32),
            pt2.astype(np.int32),
            param.heading_color,
            param.thickness
        )

        # show heading direction with a circle (easier than arrowhead)
        overlay = cv2.circle(
            overlay,
            pt2.astype(np.int32),
            2,
            param.heading_color,
            param.thickness
        )
    
        return overlay