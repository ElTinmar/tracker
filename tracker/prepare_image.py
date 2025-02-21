from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import cv2

def crop(
        image: NDArray,
        source_crop_dimension_px: Tuple[int, int],
        vertical_offset_px: int = 0,
        centroid: Optional[NDArray] = None,
    ) -> Optional[Tuple[NDArray, NDArray]]:

    # TODO make sure this is ok
    if centroid is None:
        centroid = np.array(image.shape) // 2
    
    # crop to get fixed image size 
    w, h = source_crop_dimension_px
    origin = np.asarray((-w//2, -h//2+vertical_offset_px))
    left, bottom = centroid.astype(np.int32) + origin 
    right, top = left+w, bottom+h

    pad_left = max(0, -left)
    pad_right = max(0, right - image.shape[1])
    pad_bottom = max(0, -bottom)
    pad_top = max(0, top - image.shape[0])

    if (bottom+pad_bottom >= top-pad_top) or (left+pad_left >= right-pad_right):
        return None
    
    image_crop = np.zeros((h, w), dtype=image.dtype)
    image_crop[pad_bottom:h-pad_top, pad_left:w-pad_right] = image[bottom+pad_bottom:top-pad_top, left+pad_left:right-pad_right]
    
    # TODO is this still necessary?
    if image_crop.size == 0:
        return None

    return (origin, image_crop)

def resize(
        image: NDArray,
        target_crop_dimension_px: Tuple[int, int], 
    ) -> NDArray:

    # resize image
    image_resized = cv2.resize(
        image, 
        target_crop_dimension_px, 
        interpolation=cv2.INTER_NEAREST
    )
    return image_resized

