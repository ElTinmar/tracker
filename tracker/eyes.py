from dataclasses import dataclass
import cv2
from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple, Dict, Optional
from image_tools import bwareafilter_props, bwareafilter, enhance, im2uint8, im2rgb
from geometry import ellipse_direction, angle_between_vectors, col_to_row, to_homogeneous, from_homogeneous
from tracker import Tracker, TrackingOverlay

@dataclass
class EyesTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    eye_brightness: float = 0.2
    eye_gamma: float = 1.0
    eye_dyntresh_res: int = 20
    eye_contrast: float = 1.0
    eye_size_lo_mm: float = 1.0
    eye_size_hi_mm: float = 10.0
    blur_sz_mm: float = 0.05
    median_filter_sz_mm: float = 0.15
    crop_dimension_mm: Tuple[float, float] = (1.2, 1.2) 
    crop_offset_mm: float = -0.3

    def mm2px(self, val_mm):
        return int(val_mm * self.target_pix_per_mm) 
    
    @property
    def resize(self):
        return self.target_pix_per_mm/self.pix_per_mm
    
    @property
    def eye_size_lo_px(self):
        return self.mm2px(self.eye_size_lo_mm)
    
    @property
    def eye_size_hi_px(self):
        return self.mm2px(self.eye_size_hi_mm)
    
    @property
    def blur_sz_px(self):
        return self.mm2px(self.blur_sz_mm)
    
    @property
    def median_filter_sz_px(self):
        return self.mm2px(self.median_filter_sz_mm)
    
    @property
    def crop_dimension_px(self):
        return (
            self.mm2px(self.crop_dimension_mm[0]),
            self.mm2px(self.crop_dimension_mm[1])
        ) 
    
    @property
    def crop_offset_px(self):
        return self.mm2px(self.crop_offset_mm)

@dataclass
class EyesTrackerParamOverlay:
    pix_per_mm: float = 40.0
    eye_len_mm: float = 0.2
    color_eye_left: tuple = (255, 255, 128)
    color_eye_right: tuple = (128, 255, 255)
    thickness: int = 2

    def mm2px(self, val_mm):
        val_px = int(val_mm * self.pix_per_mm) 
        return val_px
    
    @property
    def eye_len_px(self):
        return self.mm2px(self.eye_len_mm)
        
@dataclass
class EyesTracking:
    centroid: NDArray
    left_eye: dict
    right_eye: dict
    mask: NDArray
    image: NDArray
    
    def to_csv(self):
        '''export data as csv'''
        pass

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

def disp_eye(
        image: NDArray, 
        eye_centroid: NDArray,
        eye_direction: NDArray,
        transformation_matrix: NDArray,
        color: tuple, 
        eye_len_px: float, 
        thickness: int
    ) -> NDArray:

    overlay = image.copy()

    # draw two lines from eye centroid 
    pt1 = eye_centroid
    pt2 = pt1 + eye_len_px * eye_direction
    pt3 = pt1 - eye_len_px * eye_direction

    # compute transformation
    pts = np.vstack((pt1, pt2, pt3))
    pts_ = from_homogeneous((transformation_matrix @ to_homogeneous(pts).T).T)

    overlay = cv2.line(
        overlay,
        pts_[0].astype(np.int32),
        pts_[1].astype(np.int32),
        color,
        thickness
    )
    
    overlay = cv2.line(
        overlay,
        pts_[0].astype(np.int32),
        pts_[2].astype(np.int32),
        color,
        thickness
    )

    # indicate eye direction with a circle (easier than arrowhead)
    overlay = cv2.circle(
        overlay,
        pts_[2].astype(np.int32),
        2,
        color,
        thickness
    )

    return overlay

class EyesTracker(Tracker):

    def __init__(
            self, 
            tracking_param: EyesTrackerParamTracking, 
        ) -> None:

        self.tracking_param = tracking_param

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
            left_eye = left_eye,
            right_eye = right_eye,
            mask = im2uint8(mask),
            image = im2uint8(image_crop)
        )

        return res
    
class EyesOverlay(TrackingOverlay):

    def __init__(
            self, 
            overlay_param: EyesTrackerParamOverlay
        ) -> None:

        self.overlay_param = overlay_param

    def overlay(
            self,
            image: NDArray, 
            tracking: Optional[EyesTracking], 
            transformation_matrix: NDArray,
        ) -> Optional[NDArray]:

        '''
        Coordinate system: origin = fish centroid, rotation = fish heading
        '''

        if tracking is not None:

            overlay = im2rgb(image)
            
            # left eye
            if tracking.left_eye is not None:

                overlay = disp_eye(
                    overlay, 
                    tracking.left_eye['centroid'],
                    tracking.left_eye['direction'],
                    transformation_matrix,
                    self.overlay_param.color_eye_left, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )

            # right eye
            if tracking.right_eye is not None:   

                overlay = disp_eye(
                    overlay, 
                    tracking.right_eye['centroid'],
                    tracking.right_eye['direction'],
                    transformation_matrix,
                    self.overlay_param.color_eye_right, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )
        
            return overlay

class EyesTrackerGPU(Tracker):

    def track(
            image: NDArray, 
            centroid: Optional[NDArray],
        ) -> Optional[EyesTracking]:
        '''TODO'''