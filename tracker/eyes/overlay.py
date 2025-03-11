import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict
from image_tools import im2uint8, im2rgb
from geometry import transform_point_2d, Affine2DTransform
from .core import EyesOverlay

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
    pts_ = transform_point_2d(transformation_matrix, pts)

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

    def overlay_global(
            self,
            image: NDArray, 
            tracking: Optional[NDArray],
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> Optional[NDArray]:

        if tracking is None:
            return None
        
        centroid = {
            'left_eye': tracking['left_eye']['centroid_global'],
            'right_eye': tracking['right_eye']['centroid_global'],
        }

        direction = {
            'left_eye': tracking['left_eye']['direction_global'],
            'right_eye': tracking['right_eye']['direction_global'],
        }
            
        return self._overlay(
            centroid = centroid,
            direction = direction,
            image = image,
            transformation_matrix = transformation_matrix
        )
    
    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:

        if tracking is None:
            return None

        centroid = {
            'left_eye': tracking['left_eye']['centroid_cropped'],
            'right_eye': tracking['right_eye']['centroid_cropped'],
        }

        direction = {
            'left_eye': tracking['left_eye']['direction'],
            'right_eye': tracking['right_eye']['direction'],
        }

        return self._overlay(
            centroid = centroid,
            direction = direction,
            image = tracking['image_cropped'],
        )

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        
        if tracking is None:
            return None

        centroid = {
            'left_eye': tracking['left_eye']['centroid_resized'],
            'right_eye': tracking['right_eye']['centroid_resized'],
        }

        direction = {
            'left_eye': tracking['left_eye']['direction'],
            'right_eye': tracking['right_eye']['direction'],
        }

        return self._overlay(
            centroid = centroid,
            direction = direction,
            image = tracking['image_processed'],
        )
    
    def _overlay(
            self,
            image: NDArray, 
            centroid: Dict[str, NDArray],
            direction: Dict[str, NDArray],
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> NDArray:

        '''
        Coordinate system: origin = fish centroid, rotation = fish heading
        '''

        overlay = im2rgb(im2uint8(image))
        original = overlay.copy()        
        
        it = zip(
            ['left_eye', 'right_eye'], 
            [self.overlay_param.color_left_BGR, self.overlay_param.color_right_BGR]
        )

        for eye, col in it:
            overlay = disp_eye(
                overlay, 
                centroid[eye],
                direction[eye],
                transformation_matrix,
                col, 
                self.overlay_param.eye_len_px, 
                self.overlay_param.thickness,
                self.overlay_param.arrow_radius_px
            )

        overlay = cv2.addWeighted(
            overlay, self.overlay_param.alpha, 
            original, 1 - self.overlay_param.alpha, 
            0)
        
        return overlay
