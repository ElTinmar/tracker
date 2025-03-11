from typing import NamedTuple, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import cv2
from tracker.core import ParamTracking
from image_tools import enhance
from geometry import Affine2DTransform

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

    if image.shape[:2] == crop_dimension_px[::-1]:
        return image, Affine2DTransform.identity()

    if centroid is None:
        centroid = np.array(image.shape[:2]) // 2
        centroid = centroid[::-1] # convert to (x, y)
    
    # crop to get fixed image size 
    w, h = crop_dimension_px
    left, bottom = centroid.astype(np.int32) + np.asarray((-w//2, -h//2+vertical_offset_px)) 
    right, top = left+w, bottom+h

    pad_left = max(0, -left)
    pad_right = max(0, right - image.shape[1])
    pad_bottom = max(0, -bottom)
    pad_top = max(0, top - image.shape[0])

    if (bottom+pad_bottom >= top-pad_top) or (left+pad_left >= right-pad_right):
        return None
    
    if image.ndim == 2: 
        image_cropped = np.zeros((h, w), dtype=image.dtype)
    else: 
        image_cropped = np.zeros((h, w, image.shape[-1]), dtype=image.dtype)

    image_cropped[pad_bottom:h-pad_top, pad_left:w-pad_right] = image[
        bottom+pad_bottom:top-pad_top, 
        left+pad_left:right-pad_right
    ]

    # cropped space to input space
    crop_transform = Affine2DTransform.translation(left, bottom)

    return image_cropped, crop_transform

def resize(
        image: NDArray,
        target_dimension_px: Tuple[int, int], 
    ) -> Tuple[NDArray]:
    """
    Resize an image to the specified dimensions.

    Args:
        image (NDArray): Input image as a NumPy array (H, W, C) or (H, W).
        target_dimension_px (Tuple[int, int]): Target dimensions as (width, height).

    Returns:
        NDArray: The resized image.
    """

    if image.shape[:2] == target_dimension_px[::-1]:
        return image, Affine2DTransform.identity()

    image_resized = cv2.resize(
        image, 
        target_dimension_px, 
        interpolation=cv2.INTER_NEAREST
    )

    # resized space to input space
    sx = image.shape[1] / target_dimension_px[0]
    sy = image.shape[0] / target_dimension_px[1]
    resize_transform =  Affine2DTransform.scaling(sx, sy)

    return image_resized, resize_transform

class Preprocessing(NamedTuple):
    image_cropped: NDArray
    image_resized: NDArray
    image_processed: NDArray
    crop_transform: NDArray
    resize_transform: NDArray

def preprocess_image(
        image: NDArray, 
        centroid: NDArray, 
        params: ParamTracking
    ) -> Optional[Preprocessing]:
        
    # crop -----------------------
    cropping = crop(
        image = image,
        crop_dimension_px = params.crop_dimension_px,
        centroid = centroid
    )
    
    if cropping is None:
        return None

    image_cropped, crop_transform = cropping

    # resize ---------------------
    image_resized, resize_transform = resize(
        image = image_cropped,
        target_dimension_px = params.resized_dimension_px, 
    )

    # enhance --------------------
    image_processed = enhance(
        image = image_resized,
        contrast = params.contrast,
        gamma = params.gamma,
        blur_size_px = params.blur_sz_px,
        medfilt_size_px = params.median_filter_sz_px
    )

    return Preprocessing(
        image_cropped, 
        image_resized, 
        image_processed,
        crop_transform,
        resize_transform
    )