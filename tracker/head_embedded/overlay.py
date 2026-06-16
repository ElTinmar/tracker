from typing import Optional
from numpy.typing import NDArray
from image_tools import im2uint8, im2rgb
from geometry import SimilarityTransform2D
from .core import HeadEmbeddedOverlay

class HeadEmbeddedOverlay_opencv(HeadEmbeddedOverlay):

    def overlay_cropped(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_processed(self, tracking: Optional[NDArray]) -> Optional[NDArray]:
        pass

    def overlay_global(
            self, 
            image: NDArray, 
            tracking: Optional[NDArray], 
            T_global_to_input: SimilarityTransform2D = SimilarityTransform2D.identity()
        ) -> Optional[NDArray]:

        if (tracking is not None):

            overlay = im2rgb(im2uint8(image))
            overlay = self.overlay_param.tail.overlay_global(
                overlay, 
                tracking['tail'],
                T_global_to_input
            )

            return overlay
