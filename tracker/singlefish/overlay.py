from typing import Optional
from numpy.typing import NDArray
from image_tools import im2uint8, im2rgb
from geometry import SimilarityTransform2D
from .core import SingleFishOverlay

class SingleFishOverlay_opencv(SingleFishOverlay):

    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_global(
            self, 
            image: NDArray, 
            tracking: Optional[NDArray], 
            T_global_to_input: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> NDArray:

        if (tracking is not None):

            overlay = im2rgb(im2uint8(image))
            overlay = self.overlay_param.animal.overlay_global(
                overlay, 
                tracking['animals'], 
                T_global_to_input
            )         

            if (self.overlay_param.body is not None) and ('body' in tracking.dtype.fields):
                
                overlay = self.overlay_param.body.overlay_global(
                    overlay, 
                    tracking['body'], 
                    T_global_to_input
                )

                if (self.overlay_param.eyes is not None) and ('eyes' in tracking.dtype.fields):
                    overlay = self.overlay_param.eyes.overlay_global(
                        overlay, 
                        tracking['eyes'],
                        T_global_to_input
                    )
                
                if (self.overlay_param.tail is not None) and ('tail' in tracking.dtype.fields):
                    overlay = self.overlay_param.tail.overlay_global(
                        overlay, 
                        tracking['tail'],
                        T_global_to_input
                    )

            return overlay
