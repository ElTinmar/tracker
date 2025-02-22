from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple, Optional
from image_tools import bwareafilter_props_cv2, bwareafilter_cv2
from geometry import angle_between_vectors
from .core import DTYPE_EYE
from geometry import transform2d
import cv2

def get_eye_prop_cv2(
        centroid: NDArray, 
        principal_axis: Optional[NDArray], 
        origin: NDArray, 
        resize: float,
        transformation_matrix: NDArray
    ) -> Optional[NDArray]:

    if principal_axis is None:
        return None

    # fish must be vertical head up
    heading = np.array([0, 1], dtype=np.single)

    eye_angle = angle_between_vectors(principal_axis, heading)
    eye_centroid = centroid + origin 
    eye_dir_original_space = transform2d(transformation_matrix, principal_axis)
    eye_centroid_original_space = transform2d(transformation_matrix, eye_centroid/resize)
    eye =  np.array(
        (
            principal_axis, 
            eye_angle, 
            eye_centroid/resize,
            eye_dir_original_space,
            eye_centroid_original_space
        ),
        dtype = DTYPE_EYE
    )
    return eye

def assign_features(blob_centroids: ArrayLike) -> Tuple[int, int, int]:
    """From Duncan, returns indices of swimbladder, left eye and right eye"""
    
    centroids = np.asarray(blob_centroids)
    
    # find swimbladder
    distances = pdist(centroids)
    swimbladder_index = 2 - np.argmin(distances)

    # find eyes
    eye_indices = [i for i in range(3) if i != swimbladder_index]
    
    # Getting left and right eyes
    # NOTE: numpy automatically adds 0 to 3rd dimension when 
    # computing cross-product if input arrays are 2D.
    eye_vectors = centroids[eye_indices] - centroids[swimbladder_index]
    cross_product = np.cross(*eye_vectors)
    if cross_product < 0:
        eye_indices = eye_indices[::-1]
    left_eye_index, right_eye_index = eye_indices

    return swimbladder_index, left_eye_index, right_eye_index

def find_eyes_and_swimbladder(
        image: NDArray, 
        eye_dyntresh_res: int, 
        eye_size_lo_px: float, 
        eye_size_hi_px: float,
        thresh_lo: float = 0,
        thresh_hi: float = 1,
    ) -> Tuple:
    
    # OPTIM this is slow
    thresholds = np.linspace(thresh_lo,thresh_hi,eye_dyntresh_res)
    found_eyes_and_sb = False
    for t in thresholds:

        # actual tracking starts here
        mask = cv2.compare(image, t, cv2.CMP_GT)
        props = bwareafilter_props_cv2(
            mask, 
            min_size = eye_size_lo_px, 
            max_size = eye_size_hi_px
        )
        if len(props) == 3:
            found_eyes_and_sb = True
            mask = bwareafilter_cv2(
                mask, 
                min_size = eye_size_lo_px, 
                max_size = eye_size_hi_px
            )
            break

    return (found_eyes_and_sb, props, mask)

