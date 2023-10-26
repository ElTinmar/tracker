from dataclasses import dataclass
import cv2
from scipy.spatial.distance import pdist
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from image_tools import bwareafilter_props, bwareafilter, imcontrast
from geometry import ellipse_direction, angle_between_vectors, Rect

@dataclass
class EyesTrackerParamTracking:
    pix_per_mm: float = 40.0
    target_pix_per_mm: float = 20.0
    eye_norm: float = 0.2
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

class EyesTracker:
    def __init__(
            self, 
            tracking_param: EyesTrackerParamTracking, 
            overlay_param: EyesTrackerParamOverlay
        ) -> None:
        self.tracking_param = tracking_param
        self.overlay_param = overlay_param
    
    @staticmethod
    def get_eye_prop(blob, resize):
        # fish must be vertical head up
        heading = np.array([0, 1], dtype=np.float32)

        eye_dir = ellipse_direction(blob.inertia_tensor, heading)
        eye_angle = angle_between_vectors(eye_dir, heading)
        # (row,col) to (x,y) coordinates 
        y, x = blob.centroid
        eye_centroid = np.array([x, y],dtype = np.float32)
        return {'direction': eye_dir, 'angle': eye_angle, 'centroid': eye_centroid/resize}
    
    @staticmethod
    def assign_features(blob_centroids):
            """From Duncan, returns indices of swimbladder, left eye and right eye"""
            centres = np.array(blob_centroids)
            distances = pdist(blob_centroids)
            sb_idx = 2 - np.argmin(distances)
            eye_idxs = [i for i in range(3) if i != sb_idx]
            eye_vectors = centres[eye_idxs] - centres[sb_idx]
            cross_product = np.cross(*eye_vectors)
            if cross_product < 0:
                eye_idxs = eye_idxs[::-1]
            left_idx, right_idx = eye_idxs
            return sb_idx, left_idx, right_idx
    
    @staticmethod 
    def find_eyes_and_swimbladder(image, eye_dyntresh_res, eye_size_lo_px, eye_size_hi_px):
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
    
    def get_roi_coords(self, centroid):
        w, h = self.tracking_param.crop_dimension_px
        left, bottom = centroid * self.tracking_param.resize
        left = left - w//2
        bottom = bottom - h//2 + self.tracking_param.crop_offset_px
        return int(left), int(bottom), w, h
     
    def track(self, image: NDArray, centroid: NDArray):

        if (image is None) or (image.size == 0):
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
        left, bottom, w, h = self.get_roi_coords(centroid)
        right = left + w
        top = bottom + h
        image_crop = image[bottom:top, left:right]
        if image_crop.size == 0:
            return None

        # tune image contrast and gamma
        image_crop = imcontrast(
            image_crop,
            self.tracking_param.eye_contrast,
            self.tracking_param.eye_gamma,
            self.tracking_param.eye_norm,
            self.tracking_param.blur_sz_px,
            self.tracking_param.median_filter_sz_px
        )

        # sweep threshold to obtain 3 connected component within size range (include SB)
        found_eyes_and_sb, props, mask = self.find_eyes_and_swimbladder(
            image_crop, 
            self.tracking_param.eye_dyntresh_res, 
            self.tracking_param.eye_size_lo_px, 
            self.tracking_param.eye_size_hi_px
        )
        
        if found_eyes_and_sb: 
            # identify left eye, right eye and swimbladder
            blob_centroids = np.array([blob.centroid for blob in props])
            sb_idx, left_idx, right_idx = self.assign_features(blob_centroids)

            # compute eye orientation
            left_eye = self.get_eye_prop(props[left_idx], self.tracking_param.resize)
            right_eye = self.get_eye_prop(props[right_idx], self.tracking_param.resize)
            #new_heading = (props[left_idx].centroid + props[right_idx].centroid)/2 - props[sb_idx].centroid
            #new_heading = new_heading / np.linalg.norm(new_heading)

        res = EyesTracking(
            centroid = centroid,
            left_eye = left_eye,
            right_eye = right_eye,
            mask = (255*mask).astype(np.uint8),
            image = (255*image_crop).astype(np.uint8)
        )
        return res

    @staticmethod
    def disp_eye(
            image: NDArray, 
            eye_centroid: NDArray,
            eye_direction: NDArray,
            color: tuple, 
            eye_len_px: float, 
            thickness: int
        ) -> NDArray:

        pt1 = eye_centroid
        pt2 = pt1 + eye_len_px * eye_direction
        image = cv2.line(
            image,
            pt1.astype(np.int32),
            pt2.astype(np.int32),
            color,
            thickness
        )
        pt2 = pt1 - eye_len_px * eye_direction
        image = cv2.line(
            image,
            pt1.astype(np.int32),
            pt2.astype(np.int32),
            color,
            thickness
        )
        image = cv2.circle(
            image,
            pt2.astype(np.int32),
            2,
            color,
            thickness
        )
        return image
    
    def overlay(
            self, 
            image: NDArray, 
            tracking: EyesTracking, 
            translation_vec: NDArray,
            rotation_mat: NDArray
        ) -> NDArray:

        if tracking is not None:

            left, bottom, _, _ = self.get_roi_coords(tracking.centroid)

            if tracking.left_eye is not None:
                image = self.disp_eye(
                    image, 
                    rotation_mat @ (tracking.left_eye['centroid'] + np.array((left, bottom))/self.tracking_param.resize - tracking.centroid) + translation_vec,
                    rotation_mat @ tracking.left_eye['direction'],
                    self.overlay_param.color_eye_left, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )
            if tracking.right_eye is not None:   
                image = self.disp_eye(
                    image, 
                    rotation_mat @ (tracking.right_eye['centroid'] + np.array((left, bottom))/self.tracking_param.resize - tracking.centroid) + translation_vec,
                    rotation_mat @ tracking.right_eye['direction'],
                    self.overlay_param.color_eye_right, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )
        
        return image

    def overlay_local(self, tracking: EyesTracking):
        image = None
        if tracking is not None:
            image = tracking.image.copy()
            image = np.dstack((image,image,image))
            if tracking.left_eye is not None:
                image = self.disp_eye(
                    image, 
                    tracking.left_eye['centroid'] * self.tracking_param.resize,
                    tracking.left_eye['direction'],
                    self.overlay_param.color_eye_left, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )
            if tracking.right_eye is not None:   
                image = self.disp_eye(
                    image, 
                    tracking.right_eye['centroid'] * self.tracking_param.resize,
                    tracking.right_eye['direction'],
                    self.overlay_param.color_eye_right, 
                    self.overlay_param.eye_len_px, 
                    self.overlay_param.thickness
                )
        
        return image