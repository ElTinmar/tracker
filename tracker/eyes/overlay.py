import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import im2uint8, im2rgb
from geometry import to_homogeneous, from_homogeneous, Affine2DTransform
from .core import EyesOverlay, EyesTracking

def disp_eye(
        image: NDArray, 
        eye_centroid: NDArray,
        eye_direction: NDArray,
        transformation_matrix: NDArray,
        color: tuple, 
        eye_len_px: float, 
        thickness: int,
        radius: int
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
    filled = -1
    
    overlay = cv2.circle(
        overlay,
        pts_[2].astype(np.int32),
        radius, 
        color,
        filled
    )

    return overlay
    
class EyesOverlay_opencv(EyesOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[EyesTracking], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish centroid, rotation = fish heading
        '''

        if tracking is not None:

            overlay = im2rgb(im2uint8(image))
            original = overlay.copy()        
            
            # left eye
            if tracking.left_eye is not None and tracking.left_eye.direction is not None:

                overlay = disp_eye(
                    overlay, 
                    tracking.left_eye.centroid,
                    tracking.left_eye.direction,
                    transformation_matrix,
                    self.overlay_param.color_eye_left_BGR, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness,
                    self.overlay_param.arrow_radius_px
                )

            # right eye
            if tracking.right_eye is not None and tracking.right_eye.direction is not None:   

                overlay = disp_eye(
                    overlay, 
                    tracking.right_eye.centroid,
                    tracking.right_eye.direction,
                    transformation_matrix,
                    self.overlay_param.color_eye_right_BGR, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness,
                    self.overlay_param.arrow_radius_px
                )

            overlay = cv2.addWeighted(overlay, self.overlay_param.alpha, original, 1 - self.overlay_param.alpha, 0)
            
            return overlay
