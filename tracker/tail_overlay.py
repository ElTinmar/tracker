from dataclasses import dataclass
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .roi_coords import get_roi_coords
from .tail import TailTracking
from image_tools import im2rgb

@dataclass
class TailTrackerParamOverlay:
    pix_per_mm: float = 40
    color_tail: tuple = (255, 128, 128)
    thickness: int = 2
    
# TODO check if using polyline is faster

def overlay(
        image: NDArray, 
        tracking: Optional[TailTracking], 
        param: TailTrackerParamOverlay,
        translation_vec: NDArray,
        rotation_mat: NDArray,
        scale: float
    ) -> Optional[NDArray]:

    if tracking is not None:

        left, bottom, _, _ = get_roi_coords(tracking.centroid)
            
        if tracking.skeleton_interp is not None:

            skeleton_interp = tracking.skeleton_interp + np.array((left, bottom))/scale - tracking.centroid
            transformed_coord = (rotation_mat @ skeleton_interp.T).T + translation_vec
            tail_segments = zip(transformed_coord[:-1,], transformed_coord[1:,])
            for pt1, pt2 in tail_segments:
                image = cv2.line(
                    image,
                    pt1.astype(np.int32),
                    pt2.astype(np.int32),
                    param.color_tail,
                    param.thickness
                )
        
    return image

