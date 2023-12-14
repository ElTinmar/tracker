import cv2
from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple, Dict, Optional
from image_tools import bwareafilter_props, bwareafilter, enhance, im2uint8
from geometry import ellipse_direction, angle_between_vectors, to_homogeneous, from_homogeneous
from .core import EyesTracker, EyesTracking

def get_eye_prop(blob, offset: NDArray, resize: float) -> Dict:

    # fish must be vertical head up
    heading = np.array([0, 1], dtype=np.float32)

    eye_dir = ellipse_direction(blob.inertia_tensor, heading)
    eye_angle = angle_between_vectors(eye_dir, heading)
    # (row,col) to (x,y) coordinates 
    y, x = blob.centroid 
    eye_centroid = np.array([x, y], dtype = np.float32) + offset
    return {'direction': eye_dir, 'angle': eye_angle, 'centroid': eye_centroid/resize}


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

class EyesTracker_CPU(EyesTracker):

    def track(
            self,
            image: NDArray, 
            centroid: Optional[NDArray],
        ) -> Optional[EyesTracking]:

        if (image is None) or (image.size == 0) or (centroid is None):
            return None

        if self.tracking_param.resize != 1:
            image = cv2.resize(
                image, 
                None, 
                None,
                self.tracking_param.resize,
                self.tracking_param.resize,
                cv2.INTER_NEAREST
            )

        left_eye = None
        right_eye = None
        new_heading = None

        # crop image
        w, h = self.tracking_param.crop_dimension_px
        offset = np.array((-w//2, -h//2+self.tracking_param.crop_offset_px), dtype=np.int32)
        left, bottom = (centroid * self.tracking_param.resize).astype(np.int32) + offset 
        right, top = left+w, bottom+h 

        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = enhance(
            image_crop,
            self.tracking_param.eye_contrast,
            self.tracking_param.eye_gamma,
            self.tracking_param.eye_brightness,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # sweep threshold to obtain 3 connected component within size range (include SB)
        found_eyes_and_sb, props, mask = find_eyes_and_swimbladder(
            image_crop, 
            self.tracking_param.eye_dyntresh_res, 
            self.tracking_param.eye_size_lo_px, 
            self.tracking_param.eye_size_hi_px
        )
        
        if found_eyes_and_sb: 
            # identify left eye, right eye and swimbladder
            blob_centroids = np.array([blob.centroid for blob in props])
            sb_idx, left_idx, right_idx = assign_features(blob_centroids)

            # compute eye orientation
            left_eye = get_eye_prop(props[left_idx], offset, self.tracking_param.resize)
            right_eye = get_eye_prop(props[right_idx], offset, self.tracking_param.resize)
            #new_heading = (props[left_idx].centroid + props[right_idx].centroid)/2 - props[sb_idx].centroid
            #new_heading = new_heading / np.linalg.norm(new_heading)

        res = EyesTracking(
            centroid = centroid,
            offset = offset,
            left_eye = left_eye,
            right_eye = right_eye,
            mask = im2uint8(mask),
            image = im2uint8(image_crop)
        )

        return res
    