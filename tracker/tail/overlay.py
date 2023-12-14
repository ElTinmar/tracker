import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from image_tools import im2uint8, im2rgb
from geometry import to_homogeneous, from_homogeneous
from .core import TailOverlay, TailTracking

class TailOverlay_opencv(TailOverlay):

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[TailTracking], 
            transformation_matrix: NDArray
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish centroid, rotation = fish heading
        '''
                
        if tracking is not None:         

            overlay = im2rgb(im2uint8(image))       
            
            if tracking.skeleton_interp is not None:
                
                transformed_coord = from_homogeneous((transformation_matrix @ to_homogeneous(tracking.skeleton_interp).T).T)
                tail_segments = zip(transformed_coord[:-1,], transformed_coord[1:,])
                for pt1, pt2 in tail_segments:
                    overlay = cv2.line(
                        overlay,
                        pt1.astype(np.int32),
                        pt2.astype(np.int32),
                        self.overlay_param.color_tail,
                        self.overlay_param.thickness
                    )
            
            return overlay
