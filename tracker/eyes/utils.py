from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple, Dict
from image_tools import bwareafilter_props, bwareafilter
from geometry import ellipse_direction, angle_between_vectors
from .core import Eye

def get_eye_prop(
        centroid: NDArray, 
        inertia_tensor: NDArray, 
        offset: NDArray, 
        resize: float
    ) -> Eye:

    # fish must be vertical head up
    heading = np.array([0, 1], dtype=np.float32)

    eye_dir = ellipse_direction(inertia_tensor, heading)
    eye_angle = angle_between_vectors(eye_dir, heading)
    # (row,col) to (x,y) coordinates 
    y, x = centroid 
    eye_centroid = np.array([x, y], dtype = np.float32) + offset
    return Eye(direction=eye_dir, angle=eye_angle, centroid=eye_centroid/resize)


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
        eye_size_hi_px: float
    ) -> Tuple:
    
    # OPTIM this is slow
    thresholds = np.linspace(1/eye_dyntresh_res,1,eye_dyntresh_res)
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