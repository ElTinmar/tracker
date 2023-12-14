import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import im2uint8, im2rgb
from geometry import to_homogeneous, from_homogeneous
from .core import EyesOverlay, EyesTracking

def disp_eye(
        image: NDArray, 
        eye_centroid: NDArray,
        eye_direction: NDArray,
        transformation_matrix: NDArray,
        color: tuple, 
        eye_len_px: float, 
        thickness: int
    ) -> NDArray:

    overlay = image.copy()

    # draw two lines from eye centroid 
    pt1 = eye_centroid
    pt2 = pt1 + eye_len_px * eye_direction
    pt3 = pt1 - eye_len_px * eye_direction

    # compute transformation
    pts = np.vstack((pt1, pt2, pt3))
    pts_ = from_homogeneous((transformation_matrix @ to_homogeneous(pts).T).T)

    overlay = cv2.line(
        overlay,
        pts_[0].astype(np.int32),
        pts_[1].astype(np.int32),
        color,
        thickness
    )
    
    overlay = cv2.line(
        overlay,
        pts_[0].astype(np.int32),
        pts_[2].astype(np.int32),
        color,
        thickness
    )

    # indicate eye direction with a circle (easier than arrowhead)
    overlay = cv2.circle(
        overlay,
        pts_[2].astype(np.int32),
        2, # TODO this is hardcoded, make it a param ?
        color,
        thickness
    )

    return overlay
    
class EyesOverlay_opencv(EyesOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[EyesTracking], 
            transformation_matrix: NDArray,
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish centroid, rotation = fish heading
        '''

        if tracking is not None:

            overlay = im2rgb(im2uint8(image))
            
            # left eye
            if tracking.left_eye is not None:

                overlay = disp_eye(
                    overlay, 
                    tracking.left_eye['centroid'],
                    tracking.left_eye['direction'],
                    transformation_matrix,
                    self.overlay_param.color_eye_left, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )

            # right eye
            if tracking.right_eye is not None:   

                overlay = disp_eye(
                    overlay, 
                    tracking.right_eye['centroid'],
                    tracking.right_eye['direction'],
                    transformation_matrix,
                    self.overlay_param.color_eye_right, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )
        
            return overlay
