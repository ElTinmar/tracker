from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from image_tools import  enhance

def prepare_image(
        image: NDArray,
        source_crop_dimension_px: Tuple[int, int],
        target_crop_dimension_px: Tuple[int, int], 
        centroid: NDArray,
        contrast: float,
        gamma: float,
        brightness: float,
        blur_sz_px: float,
        median_filter_sz_px: float
    ) -> Tuple[NDArray, NDArray, NDArray]:
    '''Pad, crop, resize and enhance image before tracking'''
    
    # pad image with zeros then crop to get fixed image size 
    w, h = source_crop_dimension_px
    pad_width = np.max(source_crop_dimension_px)
    image_padded = np.pad(image, (pad_width,pad_width))

    # crop image: put centroid in the middle
    origin = np.asarray((-w//2, -h//2))
    left, bottom = centroid.astype(np.int32) + origin + np.array([pad_width,pad_width])
    right, top = left+w, bottom+h 
    image_crop = image_padded[bottom:top, left:right]
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