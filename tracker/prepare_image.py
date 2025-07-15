from typing import NamedTuple, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import cv2
from tracker.core import ParamTracking
from image_tools import enhance
from geometry import SimilarityTransform2D

class Cropped(NamedTuple):
    image_cropped: NDArray
    background_image_cropped: NDArray
    T_cropped_to_input: SimilarityTransform2D
    T_input_to_cropped: SimilarityTransform2D

def crop(
        image: NDArray,
        background_image: NDArray,
        crop_dimension_px: Tuple[int, int],
        vertical_offset_px: int = 0,
        centroid: Optional[NDArray] = None,
    ) -> Optional[Cropped]:
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

    if crop_dimension_px == (0,0) or image.shape[:2] == crop_dimension_px[::-1]:
        
        return Cropped(
            image, 
            background_image, 
            SimilarityTransform2D.identity(), 
            SimilarityTransform2D.identity()
        )

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
    
    background_image_cropped = None
    if background_image is not None:
        background_image_cropped = np.zeros_like(image_cropped)
        background_image_cropped[pad_bottom:h-pad_top, pad_left:w-pad_right] = background_image[
            bottom+pad_bottom:top-pad_top, 
            left+pad_left:right-pad_right
        ]

    T_cropped_to_input = SimilarityTransform2D.translation(left, bottom)
    T_input_to_cropped = SimilarityTransform2D.translation(-left, -bottom)
    return Cropped(
        image_cropped, 
        background_image_cropped, 
        T_cropped_to_input, 
        T_input_to_cropped
    )

class Resized(NamedTuple):
    image_resized: NDArray 
    background_image_resized: NDArray
    T_resized_to_cropped: SimilarityTransform2D 
    T_cropped_to_resized: SimilarityTransform2D

def resize(
        image: NDArray,
        background_image: NDArray,
        target_dimension_px: Tuple[int, int], 
    ) -> Resized:
    """
    Resize an image to the specified dimensions.

    Args:
        image (NDArray): Input image as a NumPy array (H, W, C) or (H, W).
        target_dimension_px (Tuple[int, int]): Target dimensions as (width, height).

    Returns:
        NDArray: The resized image.
    """

    if image.shape[:2] == target_dimension_px[::-1]:
        return Resized(
            image, 
            background_image, 
            SimilarityTransform2D.identity(), 
            SimilarityTransform2D.identity()
        )

    image_resized = cv2.resize(
        image, 
        target_dimension_px, 
        interpolation=cv2.INTER_NEAREST
    )

    background_image_resized = None
    if background_image is not None:
        background_image_resized = cv2.resize(
            background_image, 
            target_dimension_px, 
            interpolation=cv2.INTER_NEAREST
        )

    s = image.shape[1] / target_dimension_px[0]
    T_resized_to_cropped =  SimilarityTransform2D.scaling(s)
    T_cropped_to_resized =  SimilarityTransform2D.scaling(1.0/s)

    return Resized(
        image_resized, 
        background_image_resized, 
        T_resized_to_cropped, 
        T_cropped_to_resized
    )

class Preprocessing(NamedTuple):
    image_cropped: NDArray
    background_image_cropped: NDArray
    image_resized: NDArray
    background_image_resized: NDArray
    image_subtracted: NDArray
    image_processed: NDArray
    T_cropped_to_input: SimilarityTransform2D
    T_input_to_cropped: SimilarityTransform2D
    T_resized_to_cropped: SimilarityTransform2D
    T_cropped_to_resized: SimilarityTransform2D

def preprocess_image(
        image: NDArray, 
        background_image: NDArray,
        centroid: Optional[NDArray], 
        params: ParamTracking,
        background_polarity: float = -1, # Put that in parameters?
    ) -> Optional[Preprocessing]:
        
    # crop -----------------------
    cropped = crop(
        image = image,
        background_image = background_image,
        crop_dimension_px = params.crop_dimension_px,
        vertical_offset_px = params.crop_offset_y_px,
        centroid = centroid
    )
    
    if cropped is None:
        return None

    # resize ---------------------
    resized = resize(
        image = cropped.image_cropped,
        background_image = cropped.background_image_cropped,
        target_dimension_px = params.resized_dimension_px, 
    )

    # background subtraction on cropped/resized image 
    image_subtracted = resized.image_resized
    if resized.background_image_resized is not None:
        image_subtracted = np.maximum(0, background_polarity * (resized.image_resized - resized.background_image_resized))

    # enhance --------------------
    image_processed = enhance(
        image = image_subtracted,
        contrast = params.contrast,
        gamma = params.gamma,
        blur_size_px = params.blur_sz_px,
        medfilt_size_px = params.median_filter_sz_px
    )

    return Preprocessing(
        cropped.image_cropped,
        cropped.background_image_cropped, 
        resized.image_resized, 
        resized.background_image_resized, 
        image_subtracted,
        image_processed,
        cropped.T_cropped_to_input,
        cropped.T_input_to_cropped,
        resized.T_resized_to_cropped,
        resized.T_cropped_to_resized
    )