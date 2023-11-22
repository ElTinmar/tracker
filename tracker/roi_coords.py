
from numpy.typing import NDArray
from typing import Tuple

def get_roi_coords(
        centroid: NDArray, 
        crop_dimension_px: NDArray, 
        crop_offset_px: NDArray, 
        resize: float
    ) -> Tuple[int, int, int, int]:


    w, h = crop_dimension_px
    left, bottom = centroid * resize
    left = left - w//2
    bottom = bottom - h//2 + crop_offset_px
    return int(left), int(bottom), w, h