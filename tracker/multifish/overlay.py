from typing import Optional
import numpy as np
from numpy.typing import NDArray
from image_tools import im2uint8, im2rgb
from geometry import Affine2DTransform
from .core import MultiFishOverlay

class MultiFishOverlay_opencv(MultiFishOverlay):

    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_global(
            self, 
            image: NDArray, 
            tracking: Optional[NDArray], 
            transformation_matrix: NDArray = Affine2DTransform.identity()
        ) -> NDArray:

        if (tracking is not None):

            overlay = im2rgb(im2uint8(image))

            overlay = self.overlay_param.animal.overlay_global(overlay, tracking['animals'], transformation_matrix)         

            for idx, _ in enumerate(tracking['animals']['centroids_global']):

                if (self.overlay_param.body is not None) and ('body' in tracking.dtype.fields):

                    overlay = self.overlay_param.body.overlay_global(
                        overlay, 
                        tracking['body'][idx], 
                        transformation_matrix
                    )

                    if (self.overlay_param.eyes is not None) and ('eyes' in tracking.dtype.fields):

                        overlay = self.overlay_param.eyes.overlay_global(
                            overlay, 
                            tracking['eyes'][idx],
                            transformation_matrix
                        )
                    
                    if (self.overlay_param.tail is not None) and ('tail' in tracking.dtype.fields):

                        overlay = self.overlay_param.tail.overlay_global(
                            overlay, 
                            tracking['tail'][idx],
                            transformation_matrix
                        )

            return overlay
