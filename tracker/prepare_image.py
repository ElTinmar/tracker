from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from image_tools import  enhance

def prepare_image(
        image: NDArray,
        source_crop_dimension_px: Tuple[int, int],
        target_crop_dimension_px: Tuple[int, int], 
        vertical_offset_px: int,
        centroid: NDArray,
        contrast: float,
        gamma: float,
        brightness: float,
        blur_sz_px: float,
        median_filter_sz_px: float
    ) -> Tuple[NDArray, NDArray, NDArray]:
    '''crop, resize and enhance image before tracking'''
    
    # crop to get fixed image size 
    # NOTE: this may affect the distribution of pixel values on the edges
    w, h = source_crop_dimension_px
    origin = np.asarray((-w//2, -h//2+vertical_offset_px))
    left, bottom = centroid.astype(np.int32) + origin 
    right, top = left+w, bottom+h

    pad_left = 0 if left>=0 else -left
    pad_right = right-image.shape[1] if right>=image.shape[1] else 0
    pad_bottom = 0 if bottom>=0 else -bottom
    pad_top = top-image.shape[0] if top>=image.shape[0] else 0
    
    image_crop = np.zeros((h,w), dtype=image.dtype)
    image_crop[pad_bottom:h-pad_top, pad_left:w-pad_right] = image[bottom+pad_bottom:top-pad_top, left+pad_left:right-pad_right]
    if image_crop.size == 0:
        return None

    # resize image
    image_processed = cv2.resize(
        image_crop, 
        target_crop_dimension_px, 
        interpolation=cv2.INTER_NEAREST
    )

    # tune image contrast and gamma
    image_processed = enhance(
        image_processed,
        contrast,
        gamma,
        brightness,
        blur_sz_px,
        median_filter_sz_px
    )

    return (origin, image_crop, image_processed)