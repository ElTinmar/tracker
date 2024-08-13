from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple
from image_tools import bwareafilter_props, bwareafilter
from geometry import ellipse_direction, angle_between_vectors
from .core import Eye
from geometry import transform2d, Affine2DTransform


# TODO implement watershed segmentation for eye and swimmbladder 
# this may help disconnect eye and swimmbladder when they are 
# connected together and remove the need to swipe the threshold

def get_eye_prop(
        centroid: NDArray, 
        inertia_tensor: NDArray, 
        origin: NDArray, 
        resize: float,
        transformation_matrix: NDArray
    ) -> Eye:

    # fish must be vertical head up
    heading = np.array([0, 1], dtype=np.single)

    eye_dir = ellipse_direction(inertia_tensor, heading)
    eye_angle = angle_between_vectors(eye_dir, heading)
    eye_centroid = centroid + origin 
    eye_dir_original_space = transform2d(transformation_matrix, eye_dir)
    eye_centroid_original_space = transform2d(transformation_matrix, eye_centroid/resize)

    eye =  Eye(
        direction=eye_dir, 
        angle=eye_angle, 
        centroid=eye_centroid/resize,
        direction_original_space=eye_dir_original_space,
        centroid_original_space=eye_centroid_original_space
    )

    return eye


def assign_features(blob_centroids: ArrayLike) -> Tuple[int, int, int]:
    """From Duncan, returns indices of swimbladder, left eye and right eye"""
    
    centroids = np.asarray(blob_centroids)
    
    # find swimbladder
    distances = pdist(centroids)
    sb_idx = 2 - np.argmin(distances)

    # find eyes
    eye_idxs = [i for i in range(3) if i != sb_idx]
    
    # Getting left and right eyes
    # NOTE: numpy automatically adds 0 to 3rd dimension when 
    # computing cross-product if input arrays are 2D.
    eye_vectors = centroids[eye_idxs] - centroids[sb_idx]
    cross_product = np.cross(*eye_vectors)
    if cross_product < 0:
        eye_idxs = eye_idxs[::-1]
    left_idx, right_idx = eye_idxs

    return sb_idx, left_idx, right_idx

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
        mask = 1.0*(image >= t)
        props = bwareafilter_props(
            mask, 
            min_size = eye_size_lo_px, 
            max_size = eye_size_hi_px
        )
        if len(props) == 3:
            found_eyes_and_sb = True
            mask = bwareafilter(
                mask, 
                min_size = eye_size_lo_px, 
                max_size = eye_size_hi_px
            )
            break

    return (found_eyes_and_sb, props, mask)

