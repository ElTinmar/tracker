from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from image_tools import bwareafilter_props_cv2, bwareafilter_cv2
from geometry import angle_between_vectors
from .core import DTYPE_EYE
from geometry import transform2d
import cv2
from image_tools import RegionPropsLike
from tracker.prepare_image import Preprocessing

def get_eye_properties(
        prop: RegionPropsLike,
        preproc: Preprocessing,
        transformation_matrix: NDArray,
        reference_vector: NDArray
    ) -> Optional[NDArray]:

    centroid_resized = np.asarray(prop.centroid[::-1], dtype=np.float32)
    centroid_cropped = transform2d(preproc.resize_transform, centroid_resized)        
    centroid_input = transform2d(preproc.crop_transform, centroid_cropped)
    centroid_global = transform2d(transformation_matrix, centroid_input)
    
    direction = prop.principal_axis 
    if direction is None:
        return None
    
    angle = angle_between_vectors(direction, reference_vector)
    direction_global = transform2d(transformation_matrix, direction)
    angle_global = angle_between_vectors(direction, reference_vector)

    eye =  np.array(
        (
            direction, 
            direction_global,
            angle,
            angle_global,
            centroid_resized,
            centroid_cropped,
            centroid_input,
            centroid_global,
        ),
        dtype = DTYPE_EYE
    )
    return eye

def assign_features(centroids: NDArray) -> Tuple[int, int, int]:
    """From Duncan, returns indices of swimbladder, left eye and right eye"""
        
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
    eye_index, right_eye_index = eye_indices

    return swimbladder_index, eye_index, right_eye_index

def find_eyes_and_swimbladder(
        image: NDArray, 
        eye_dyntresh_res: int, 
        eye_size_lo_px: float, 
        eye_size_hi_px: float,
        thresh_lo: float = 0,
        thresh_hi: float = 1,
    ) -> Tuple[bool, List[RegionPropsLike], NDArray]:
    
    # TODO this is slow, try to optimize
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

