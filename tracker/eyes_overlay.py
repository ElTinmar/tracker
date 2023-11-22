from dataclasses import dataclass
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .eyes import EyesTracking
from .roi_coords import get_roi_coords
from image_tools import im2rgb
    
@dataclass
class EyesTrackerParamOverlay:
    pix_per_mm: float = 40.0
    eye_len_mm: float = 0.2
    color_eye_left: tuple = (255, 255, 128)
    color_eye_right: tuple = (128, 255, 255)
    thickness: int = 2

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px
    
    @property
    def eye_len_px(self):
        return self.mm2px(self.eye_len_mm)
    
def disp_eye(
        image: NDArray, 
        eye_centroid: NDArray,
        eye_direction: NDArray,
        color: tuple, 
        eye_len_px: float, 
        thickness: int
    ) -> NDArray:

    overlay = image.copy()

    # draw two lines from eye centroid 
    pt1 = eye_centroid
    pt2 = pt1 + eye_len_px * eye_direction
    overlay = cv2.line(
        overlay,
        pt1.astype(np.int32),
        pt2.astype(np.int32),
        color,
        thickness
    )
    pt2 = pt1 - eye_len_px * eye_direction
    overlay = cv2.line(
        overlay,
        pt1.astype(np.int32),
        pt2.astype(np.int32),
        color,
        thickness
    )

    # indicate eye direction with a circle (easier than arrowhead)
    overlay = cv2.circle(
        overlay,
        pt2.astype(np.int32),
        2,
        color,
        thickness
    )

    return overlay
    

def overlay(
        image: NDArray, 
        tracking: Optional[EyesTracking], 
        param: EyesTrackerParamOverlay,
        translation_vec: NDArray,
        rotation_mat: NDArray,
        scale: float
    ) -> Optional[NDArray]:

    if tracking is not None:

        overlay = im2rgb(image)

        left, bottom, _, _ = get_roi_coords(tracking.centroid)

        # left eye
        if tracking.left_eye is not None:
            overlay = disp_eye(
                overlay, 
                rotation_mat @ (
                    tracking.left_eye['centroid'] 
                    + np.array((left, bottom))/scale 
                    - tracking.centroid
                ) + translation_vec,
                rotation_mat @ tracking.left_eye['direction'],
                param.color_eye_left, 
                param.eye_len_px, 
                param.thickness
            )

        # right eye
        if tracking.right_eye is not None:   
            overlay = disp_eye(
                overlay, 
                rotation_mat @ (
                    tracking.right_eye['centroid'] 
                    + np.array((left, bottom))/scale 
                    - tracking.centroid
                ) + translation_vec,
                rotation_mat @ tracking.right_eye['direction'],
                param.color_eye_right, 
                param.eye_len_px, 
                param.thickness
            )
    
        return overlay
