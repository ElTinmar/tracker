from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import cv2

def crop(
        image: NDArray,
        crop_dimension_px: Tuple[int, int],
        vertical_offset_px: int = 0,
        centroid: Optional[NDArray] = None,
    ) -> Optional[Tuple[NDArray, NDArray]]:
    """
    Crops a fixed-size region from an image, optionally centered around a given centroid.

    Args:
        image (NDArray): Input image as a NumPy array (H, W) or (H, W, C).
        crop_dimension_px (Tuple[int, int]): Target crop size as (width, height).
        vertical_offset_px (int): Vertical offset for cropping, default is 0.
        centroid (Optional[NDArray]): (x, y) center of crop; if None, defaults to image center.

    Returns:
        Optional[Tuple[NDArray, NDArray]]: (origin, cropped image), or None if invalid crop.
    """

    # TODO make sure this is ok
    if centroid is None:
        centroid = np.array(image.shape[:2]) // 2
        centroid = centroid[::-1] # convert to (x, y)
    
    # crop to get fixed image size 
    w, h = crop_dimension_px
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

    return (origin, image_crop)

def resize(
        image: NDArray,
        target_dimension_px: Tuple[int, int], 
    ) -> NDArray:
    """
    Resize an image to the specified dimensions.

    Args:
        image (NDArray): Input image as a NumPy array (H, W, C) or (H, W).
        target_dimension_px (Tuple[int, int]): Target dimensions as (width, height).

    Returns:
        NDArray: The resized image.
    """

    if image.shape[:2] == target_dimension_px[::-1]:
        return image

    image_resized = cv2.resize(
        image, 
        target_dimension_px, 
        interpolation=cv2.INTER_NEAREST
    )
    return image_resized

def preprocess_image(
        image: NDArray, 
        centroid: NDArray, 
        params
    ) -> Optional[Tuple[NDArray, NDArray, NDArray, NDArray]]:
        
    # pre-process image: crop/resize/tune intensity
    if params.do_crop:
        cropping = crop(
            image = image,
            crop_dimension_px = params.source_crop_dimension_px,
            centroid = centroid
        )
        
        if cropping is None:
            return None
        
        origin, image_crop = cropping
    else:
        origin = np.zeros((2,)) # TODO check that 
        image_crop = image

    if params.do_resize:
        image_resized = resize(
            image = image_crop,
            target_dimension_px = params.crop_dimension_px, 
        )
    else:
        image_resized = image_crop

    if self.tracking_param.do_enhance:
        image_processed = enhance(
            image = image_resized,
            contrast = self.tracking_param.body_contrast,
            gamma = self.tracking_param.body_gamma,
            brightness = self.tracking_param.body_brightness,
            blur_size_px = self.tracking_param.blur_sz_px,
            medfilt_size_px = self.tracking_param.median_filter_sz_px
        )
    else:
        image_processed = image_resized